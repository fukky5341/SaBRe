## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.857701161


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124)
1: (-0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204)
2: (-0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664)
3: (-0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837)
4: (-0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.23 + 0.97 = 3.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8707626, upper bound: 0.8707626

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8685782
time: 0.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8662356
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8685782
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8662356

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0569339, 0.3577400, -0.1155033, 0.8715090, -0.9284430, 0.4732433
1: -0.1231019, 0.3033502, -0.1845481, 0.4175723, -0.5406743, 0.4878983
2: -0.0100022, 0.3879363, -0.0360203, 0.5334461, -0.5434483, 0.4239566
3: -0.0531300, 0.1989977, -0.0873858, 0.2886980, -0.3418280, 0.2863835
4: -0.0072941, 0.3633413, -0.0405117, 0.4983537, -0.5056479, 0.4038530

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8661198
time: 0.35 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8662356
time: 0.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0702411, 0.5589051, -0.1155033, 0.8715090, -0.9417502, 0.6744084
1: -0.1494390, 0.3938123, -0.1845481, 0.4175723, -0.5670114, 0.5783604
2: -0.0179913, 0.4957099, -0.0360203, 0.5334461, -0.5514374, 0.5317302
3: -0.0691953, 0.2536268, -0.0873858, 0.2886980, -0.3578932, 0.3410126
4: -0.0177572, 0.4660556, -0.0405117, 0.4983537, -0.5161110, 0.5065674

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8661198
time: 0.34 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8662356
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.85 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8661198
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -0.8661198, upper bound: 0.8662356
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8661198
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -0.8662356, upper bound: 0.8662356

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0569339, 0.3577400, -0.0569339, 0.3577400, -0.4146739, 0.4146739
1: -0.1231019, 0.3033502, -0.1231019, 0.3033502, -0.4264522, 0.4264522
2: -0.0100022, 0.3879363, -0.0100022, 0.3879363, -0.3979385, 0.3979385
3: -0.0531300, 0.1989977, -0.0531300, 0.1989977, -0.2521277, 0.2521277
4: -0.0072941, 0.3633413, -0.0072941, 0.3633413, -0.3706354, 0.3706354

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566280, upper bound: 0.8670727
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8566904, upper bound: 0.8566904
time: 0.32 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0569339, 0.3577400, -0.0702411, 0.5589051, -0.6158390, 0.4279811
1: -0.1231019, 0.3033502, -0.1494390, 0.3938123, -0.5169142, 0.4527892
2: -0.0100022, 0.3879363, -0.0179913, 0.4957099, -0.5057121, 0.4059277
3: -0.0531300, 0.1989977, -0.0691953, 0.2536268, -0.3067569, 0.2681929
4: -0.0072941, 0.3633413, -0.0177572, 0.4660556, -0.4733498, 0.3810985

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566280, upper bound: 0.8672161
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8566904, upper bound: 0.8569323
time: 0.32 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0702411, 0.5589051, -0.0569339, 0.3577400, -0.4279811, 0.6158390
1: -0.1494390, 0.3938123, -0.1231019, 0.3033502, -0.4527892, 0.5169142
2: -0.0179913, 0.4957099, -0.0100022, 0.3879363, -0.4059277, 0.5057121
3: -0.0691953, 0.2536268, -0.0531300, 0.1989977, -0.2681929, 0.3067569
4: -0.0177572, 0.4660556, -0.0072941, 0.3633413, -0.3810985, 0.4733498

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567128, upper bound: 0.8651360
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8611787
time: 0.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0702411, 0.5589051, -0.0702411, 0.5589051, -0.6291462, 0.6291462
1: -0.1494390, 0.3938123, -0.1494390, 0.3938123, -0.5432513, 0.5432513
2: -0.0179913, 0.4957099, -0.0179913, 0.4957099, -0.5137013, 0.5137013
3: -0.0691953, 0.2536268, -0.0691953, 0.2536268, -0.3228221, 0.3228221
4: -0.0177572, 0.4660556, -0.0177572, 0.4660556, -0.4838129, 0.4838129

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567128, upper bound: 0.8651360
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8614206
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.84 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8566280, upper bound: 0.8670727
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8566904, upper bound: 0.8566904
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8566280, upper bound: 0.8672161
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8566904, upper bound: 0.8569323
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8567128, upper bound: 0.8651360
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8611787
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8567128, upper bound: 0.8651360
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8614206

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0333226, 0.2324602, -0.0569339, 0.3577400, -0.3910626, 0.2893941
1: -0.0909861, 0.2318684, -0.1231019, 0.3033502, -0.3943363, 0.3549704
2: 0.0015018, 0.3040031, -0.0100022, 0.3879363, -0.3864345, 0.3140053
3: -0.0403259, 0.1466544, -0.0531300, 0.1989977, -0.2393236, 0.1997845
4: 0.0037138, 0.2853576, -0.0072941, 0.3633413, -0.3596275, 0.2926517

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8565884, upper bound: 0.8565884
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8565884, upper bound: 0.8566904
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0333226, 0.2324602, -0.0702411, 0.5589051, -0.5922277, 0.3027013
1: -0.0909861, 0.2318684, -0.1494390, 0.3938123, -0.4847984, 0.3813075
2: 0.0015018, 0.3040031, -0.0179913, 0.4957099, -0.4942081, 0.3219944
3: -0.0403259, 0.1466544, -0.0691953, 0.2536268, -0.2939528, 0.2158497
4: 0.0037138, 0.2853576, -0.0177572, 0.4660556, -0.4623418, 0.3031148

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8610767, upper bound: 0.8567070
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8610767, upper bound: 0.8569323
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, -0.0569339, 0.3577400, -0.3922940, 0.3330519
1: -0.0969559, 0.2951658, -0.1231019, 0.3033502, -0.4003061, 0.4182677
2: 0.0003081, 0.3782512, -0.0100022, 0.3879363, -0.3876282, 0.3882534
3: -0.0443657, 0.1797998, -0.0531300, 0.1989977, -0.2433634, 0.2329298
4: 0.0016791, 0.3568531, -0.0072941, 0.3633413, -0.3616621, 0.3641473

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8610767
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8611787
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, -0.0438723, 0.2631503, -0.2529540, 0.1234270
1: -0.0599387, 0.2132123, -0.1082850, 0.2783291, -0.3382678, 0.3214973
2: 0.0147940, 0.2804018, -0.0040358, 0.3575630, -0.3427690, 0.2844376
3: -0.0251203, 0.1231913, -0.0447380, 0.1788540, -0.2039743, 0.1679293
4: 0.0179019, 0.2651857, 0.0001914, 0.3360023, -0.3181005, 0.2649942

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8610767
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8611787
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, -0.0702411, 0.5589051, -0.5934591, 0.3463590
1: -0.0969559, 0.2951658, -0.1494390, 0.3938123, -0.4907682, 0.4446048
2: 0.0003081, 0.3782512, -0.0179913, 0.4957099, -0.4954018, 0.3962426
3: -0.0443657, 0.1797998, -0.0691953, 0.2536268, -0.2979925, 0.2489951
4: 0.0016791, 0.3568531, -0.0177572, 0.4660556, -0.4643765, 0.3746104

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8611953
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, -0.0587790, 0.4612901, -0.4510938, 0.1383336
1: -0.0599387, 0.2132123, -0.1362630, 0.3728426, -0.4327812, 0.3494753
2: 0.0147940, 0.2804018, -0.0122981, 0.4689931, -0.4541991, 0.2926998
3: -0.0251203, 0.1231913, -0.0620543, 0.2362829, -0.2614032, 0.1852457
4: 0.0179019, 0.2651857, -0.0110655, 0.4416703, -0.4237684, 0.2762511

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8614201, upper bound: 0.8611953
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8614201, upper bound: 0.8614206
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.00 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8565884, upper bound: 0.8565884
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8565884, upper bound: 0.8566904
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8610767, upper bound: 0.8567070
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8610767, upper bound: 0.8569323
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8610767
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8567070, upper bound: 0.8611787
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8610767
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8569323, upper bound: 0.8611787
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8611953
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8614201, upper bound: 0.8611953
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.8614201, upper bound: 0.8614206

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0333226, 0.2324602, -0.0345540, 0.2761179, -0.3094406, 0.2670141
1: -0.0909861, 0.2318684, -0.0969559, 0.2951658, -0.3861519, 0.3288243
2: 0.0015018, 0.3040031, 0.0003081, 0.3782512, -0.3767494, 0.3036950
3: -0.0403259, 0.1466544, -0.0443657, 0.1797998, -0.2201257, 0.1910201
4: 0.0037138, 0.2853576, 0.0016791, 0.3568531, -0.3531393, 0.2836785

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8649217
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8666677
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0333226, 0.2324602, 0.0101963, 0.0795547, -0.1128773, 0.2222639
1: -0.0909861, 0.2318684, -0.0599387, 0.2132123, -0.3041984, 0.2918071
2: 0.0015018, 0.3040031, 0.0147940, 0.2804018, -0.2789000, 0.2892091
3: -0.0403259, 0.1466544, -0.0251203, 0.1231913, -0.1635173, 0.1717747
4: 0.0037138, 0.2853576, 0.0179019, 0.2651857, -0.2614719, 0.2674557

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, -0.0333226, 0.2324602, -0.2670141, 0.3094406
1: -0.0969559, 0.2951658, -0.0909861, 0.2318684, -0.3288243, 0.3861519
2: 0.0003081, 0.3782512, 0.0015018, 0.3040031, -0.3036950, 0.3767494
3: -0.0443657, 0.1797998, -0.0403259, 0.1466544, -0.1910201, 0.2201257
4: 0.0016791, 0.3568531, 0.0037138, 0.2853576, -0.2836785, 0.3531393

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564529, upper bound: 0.8645843
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8562842, upper bound: 0.8622127
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, 0.0247140, 0.0564726, -0.0910266, 0.2514039
1: -0.0969559, 0.2951658, -0.0438645, 0.0872573, -0.1842132, 0.3390303
2: 0.0003081, 0.3782512, 0.0219830, 0.1336619, -0.1333538, 0.3562681
3: -0.0443657, 0.1797998, -0.0219314, 0.0496015, -0.0939672, 0.2017311
4: 0.0016791, 0.3568531, 0.0228731, 0.1246520, -0.1229728, 0.3339800

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564529, upper bound: 0.8646102
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8562842, upper bound: 0.8623798
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, -0.0333226, 0.2324602, -0.2222639, 0.1128773
1: -0.0599387, 0.2132123, -0.0909861, 0.2318684, -0.2918071, 0.3041984
2: 0.0147940, 0.2804018, 0.0015018, 0.3040031, -0.2892091, 0.2789000
3: -0.0251203, 0.1231913, -0.0403259, 0.1466544, -0.1717747, 0.1635173
4: 0.0179019, 0.2651857, 0.0037138, 0.2853576, -0.2674557, 0.2614719

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8607461
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8565836, upper bound: 0.8573721
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, 0.0247140, 0.0564726, -0.0462763, 0.0548407
1: -0.0599387, 0.2132123, -0.0438645, 0.0872573, -0.1471960, 0.2570768
2: 0.0147940, 0.2804018, 0.0219830, 0.1336619, -0.1188679, 0.2584187
3: -0.0251203, 0.1231913, -0.0219314, 0.0496015, -0.0747218, 0.1451227
4: 0.0179019, 0.2651857, 0.0228731, 0.1246520, -0.1067501, 0.2423126

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8607461
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8565837, upper bound: 0.8573721
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, -0.0345540, 0.2761179, -0.3106719, 0.3106719
1: -0.0969559, 0.2951658, -0.0969559, 0.2951658, -0.3921217, 0.3921217
2: 0.0003081, 0.3782512, 0.0003081, 0.3782512, -0.3779431, 0.3779431
3: -0.0443657, 0.1797998, -0.0443657, 0.1797998, -0.2241655, 0.2241655
4: 0.0016791, 0.3568531, 0.0016791, 0.3568531, -0.3551740, 0.3551740

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575453, upper bound: 0.8645797
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8622081
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0345540, 0.2761179, 0.0101963, 0.0795547, -0.1141087, 0.2659216
1: -0.0969559, 0.2951658, -0.0599387, 0.2132123, -0.3101682, 0.3551045
2: 0.0003081, 0.3782512, 0.0147940, 0.2804018, -0.2800937, 0.3634572
3: -0.0443657, 0.1797998, -0.0251203, 0.1231913, -0.1675570, 0.2049201
4: 0.0016791, 0.3568531, 0.0179019, 0.2651857, -0.2635065, 0.3389513

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575453, upper bound: 0.8646263
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, -0.0345540, 0.2761179, -0.2659216, 0.1141087
1: -0.0599387, 0.2132123, -0.0969559, 0.2951658, -0.3551045, 0.3101682
2: 0.0147940, 0.2804018, 0.0003081, 0.3782512, -0.3634572, 0.2800937
3: -0.0251203, 0.1231913, -0.0443657, 0.1797998, -0.2049201, 0.1675570
4: 0.0179019, 0.2651857, 0.0016791, 0.3568531, -0.3389513, 0.2635065

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8607497
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0101963, 0.0795547, 0.0101963, 0.0795547, -0.0693584, 0.0693584
1: -0.0599387, 0.2132123, -0.0599387, 0.2132123, -0.2731510, 0.2731510
2: 0.0147940, 0.2804018, 0.0147940, 0.2804018, -0.2656078, 0.2656078
3: -0.0251203, 0.1231913, -0.0251203, 0.1231913, -0.1483116, 0.1483116
4: 0.0179019, 0.2651857, 0.0179019, 0.2651857, -0.2472838, 0.2472838

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8607497
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
time: 0.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.94 seconds
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8649217
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8666677
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8564529, upper bound: 0.8645843
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8562842, upper bound: 0.8622127
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8564529, upper bound: 0.8646102
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8562842, upper bound: 0.8623798
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8607461
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8565836, upper bound: 0.8573721
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8607461
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8565837, upper bound: 0.8573721
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8575453, upper bound: 0.8645797
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8622081
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8575453, upper bound: 0.8646263
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8607497
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8607497
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, -0.0345540, 0.2761179, -0.3120321, 0.3156705
1: -0.0888036, 0.2224801, -0.0969559, 0.2951658, -0.3839694, 0.3194360
2: -0.0024124, 0.2897211, 0.0003081, 0.3782512, -0.3806636, 0.2894130
3: -0.0441975, 0.1421833, -0.0443657, 0.1797998, -0.2239973, 0.1865490
4: -0.0039181, 0.2695846, 0.0016791, 0.3568531, -0.3607712, 0.2679055

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, -0.0345540, 0.2761179, -0.2892494, 0.1842596
1: -0.0775794, 0.2161378, -0.0969559, 0.2951658, -0.3727452, 0.3130937
2: 0.0068115, 0.2863497, 0.0003081, 0.3782512, -0.3714397, 0.2860416
3: -0.0347106, 0.1325780, -0.0443657, 0.1797998, -0.2145104, 0.1769437
4: 0.0104636, 0.2701671, 0.0016791, 0.3568531, -0.3463895, 0.2684880

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, 0.0101963, 0.0795547, -0.1154689, 0.2709202
1: -0.0888036, 0.2224801, -0.0599387, 0.2132123, -0.3020159, 0.2824188
2: -0.0024124, 0.2897211, 0.0147940, 0.2804018, -0.2828141, 0.2749271
3: -0.0441975, 0.1421833, -0.0251203, 0.1231913, -0.1673888, 0.1673036
4: -0.0039181, 0.2695846, 0.0179019, 0.2651857, -0.2691037, 0.2516827

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, 0.0101963, 0.0795547, -0.0926861, 0.1395093
1: -0.0775794, 0.2161378, -0.0599387, 0.2132123, -0.2907917, 0.2760765
2: 0.0068115, 0.2863497, 0.0147940, 0.2804018, -0.2735902, 0.2715557
3: -0.0347106, 0.1325780, -0.0251203, 0.1231913, -0.1579019, 0.1576983
4: 0.0104636, 0.2701671, 0.0179019, 0.2651857, -0.2547220, 0.2522652

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0333226, 0.2324602, -0.2687921, 0.3017181
1: -0.0901904, 0.2714047, -0.0909861, 0.2318684, -0.3220588, 0.3623908
2: -0.0009017, 0.3494731, 0.0015018, 0.3040031, -0.3049048, 0.3479713
3: -0.0441245, 0.1647791, -0.0403259, 0.1466544, -0.1907789, 0.2051051
4: -0.0009160, 0.3274141, 0.0037138, 0.2853576, -0.2862736, 0.3237003

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8648512, upper bound: 0.8622262
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0333226, 0.2324602, -0.2448985, 0.2174062
1: -0.0840206, 0.2807598, -0.0909861, 0.2318684, -0.3158891, 0.3717459
2: 0.0052519, 0.3607540, 0.0015018, 0.3040031, -0.2987511, 0.3592522
3: -0.0381413, 0.1672755, -0.0403259, 0.1466544, -0.1847957, 0.2076015
4: 0.0080186, 0.3421003, 0.0037138, 0.2853576, -0.2773390, 0.3383865

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8622262
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0247140, 0.0564726, -0.0928045, 0.2436814
1: -0.0901904, 0.2714047, -0.0438645, 0.0872573, -0.1774476, 0.3152692
2: -0.0009017, 0.3494731, 0.0219830, 0.1336619, -0.1345636, 0.3274901
3: -0.0441245, 0.1647791, -0.0219314, 0.0496015, -0.0937261, 0.1867105
4: -0.0009160, 0.3274141, 0.0228731, 0.1246520, -0.1255679, 0.3045410

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8560938, upper bound: 0.8646081
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0247140, 0.0564726, -0.0689109, 0.1593695
1: -0.0840206, 0.2807598, -0.0438645, 0.0872573, -0.1712779, 0.3246243
2: 0.0052519, 0.3607540, 0.0219830, 0.1336619, -0.1284100, 0.3387709
3: -0.0381413, 0.1672755, -0.0219314, 0.0496015, -0.0877429, 0.1892069
4: 0.0080186, 0.3421003, 0.0228731, 0.1246520, -0.1166334, 0.3192272

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561654, upper bound: 0.8623798
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8562474, upper bound: 0.8622517
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0333226, 0.2324602, -0.2144993, 0.1045973
1: -0.0531569, 0.1766041, -0.0909861, 0.2318684, -0.2850253, 0.2675902
2: 0.0167517, 0.2375837, 0.0015018, 0.3040031, -0.2872514, 0.2360819
3: -0.0251918, 0.0982481, -0.0403259, 0.1466544, -0.1718462, 0.1385740
4: 0.0184394, 0.2241397, 0.0037138, 0.2853576, -0.2669182, 0.2204259

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573052
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573856
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, 0.0247140, 0.0564726, -0.0385117, 0.0465607
1: -0.0531569, 0.1766041, -0.0438645, 0.0872573, -0.1404141, 0.2204686
2: 0.0167517, 0.2375837, 0.0219830, 0.1336619, -0.1169102, 0.2156007
3: -0.0251918, 0.0982481, -0.0219314, 0.0496015, -0.0747934, 0.1201795
4: 0.0184394, 0.2241397, 0.0228731, 0.1246520, -0.1062125, 0.2012666

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607461
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607461
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0345540, 0.2761179, -0.3124499, 0.3029494
1: -0.0901904, 0.2714047, -0.0969559, 0.2951658, -0.3853561, 0.3683606
2: -0.0009017, 0.3494731, 0.0003081, 0.3782512, -0.3791529, 0.3491650
3: -0.0441245, 0.1647791, -0.0443657, 0.1797998, -0.2239243, 0.2091448
4: -0.0009160, 0.3274141, 0.0016791, 0.3568531, -0.3577691, 0.3257350

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8622145
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0345540, 0.2761179, -0.2885562, 0.2186375
1: -0.0840206, 0.2807598, -0.0969559, 0.2951658, -0.3791864, 0.3777157
2: 0.0052519, 0.3607540, 0.0003081, 0.3782512, -0.3729993, 0.3604459
3: -0.0381413, 0.1672755, -0.0443657, 0.1797998, -0.2179411, 0.2116413
4: 0.0080186, 0.3421003, 0.0016791, 0.3568531, -0.3488345, 0.3404212

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622145, upper bound: 0.8621458
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622145, upper bound: 0.8622145
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0101963, 0.0795547, -0.1158866, 0.2581991
1: -0.0901904, 0.2714047, -0.0599387, 0.2132123, -0.3034027, 0.3313434
2: -0.0009017, 0.3494731, 0.0147940, 0.2804018, -0.2813035, 0.3346791
3: -0.0441245, 0.1647791, -0.0251203, 0.1231913, -0.1673158, 0.1898994
4: -0.0009160, 0.3274141, 0.0179019, 0.2651857, -0.2661016, 0.3095122

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8618827
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0101963, 0.0795547, -0.0919930, 0.1738872
1: -0.0840206, 0.2807598, -0.0599387, 0.2132123, -0.2972329, 0.3406985
2: 0.0052519, 0.3607540, 0.0147940, 0.2804018, -0.2751498, 0.3459600
3: -0.0381413, 0.1672755, -0.0251203, 0.1231913, -0.1613326, 0.1923958
4: 0.0080186, 0.3421003, 0.0179019, 0.2651857, -0.2571671, 0.3241985

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8618827
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0345540, 0.2761179, -0.2581570, 0.1058287
1: -0.0531569, 0.1766041, -0.0969559, 0.2951658, -0.3483226, 0.2735600
2: 0.0167517, 0.2375837, 0.0003081, 0.3782512, -0.3614995, 0.2372756
3: -0.0251918, 0.0982481, -0.0443657, 0.1797998, -0.2049916, 0.1426138
4: 0.0184394, 0.2241397, 0.0016791, 0.3568531, -0.3384137, 0.2224606

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8625160, upper bound: 0.8573767
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8625160, upper bound: 0.8573767
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, 0.0101963, 0.0795547, -0.0615938, 0.0610784
1: -0.0531569, 0.1766041, -0.0599387, 0.2132123, -0.2663692, 0.2365428
2: 0.0167517, 0.2375837, 0.0147940, 0.2804018, -0.2636501, 0.2227897
3: -0.0251918, 0.0982481, -0.0251203, 0.1231913, -0.1483831, 0.1233684
4: 0.0184394, 0.2241397, 0.0179019, 0.2651857, -0.2467462, 0.2062378

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.01 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573856, upper bound: 0.8668169
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8648512, upper bound: 0.8622262
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8622262
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8560938, upper bound: 0.8646081
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8561654, upper bound: 0.8623798
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8562474, upper bound: 0.8622517
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573052
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573856
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607461
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607461
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8622145
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622145, upper bound: 0.8621458
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8622145, upper bound: 0.8622145
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8618827
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8618827
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8623798
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8625160, upper bound: 0.8573767
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8625160, upper bound: 0.8573767
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.01
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8573757

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, -0.0363319, 0.2683954, -0.3043096, 0.3174485
1: -0.0888036, 0.2224801, -0.0901904, 0.2714047, -0.3602083, 0.3126704
2: -0.0024124, 0.2897211, -0.0009017, 0.3494731, -0.3518855, 0.2906228
3: -0.0441975, 0.1421833, -0.0441245, 0.1647791, -0.2089766, 0.1863078
4: -0.0039181, 0.2695846, -0.0009160, 0.3274141, -0.3313321, 0.2705006

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8462533
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8649227
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, -0.0124383, 0.1840835, -0.2199977, 0.2935548
1: -0.0888036, 0.2224801, -0.0840206, 0.2807598, -0.3695634, 0.3065007
2: -0.0024124, 0.2897211, 0.0052519, 0.3607540, -0.3631663, 0.2844691
3: -0.0441975, 0.1421833, -0.0381413, 0.1672755, -0.2114730, 0.1803246
4: -0.0039181, 0.2695846, 0.0080186, 0.3421003, -0.3460184, 0.2615660

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8462533
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, -0.0363319, 0.2683954, -0.2815269, 0.1860375
1: -0.0775794, 0.2161378, -0.0901904, 0.2714047, -0.3489841, 0.3063282
2: 0.0068115, 0.2863497, -0.0009017, 0.3494731, -0.3426616, 0.2872514
3: -0.0347106, 0.1325780, -0.0441245, 0.1647791, -0.1994897, 0.1767025
4: 0.0104636, 0.2701671, -0.0009160, 0.3274141, -0.3169505, 0.2710831

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8658886
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, -0.0124383, 0.1840835, -0.1972150, 0.1621439
1: -0.0775794, 0.2161378, -0.0840206, 0.2807598, -0.3583393, 0.3001584
2: 0.0068115, 0.2863497, 0.0052519, 0.3607540, -0.3539424, 0.2810978
3: -0.0347106, 0.1325780, -0.0381413, 0.1672755, -0.2019861, 0.1707193
4: 0.0104636, 0.2701671, 0.0080186, 0.3421003, -0.3316367, 0.2621485

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8658886
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, 0.0179609, 0.0712747, -0.1071889, 0.2631557
1: -0.0888036, 0.2224801, -0.0531569, 0.1766041, -0.2654077, 0.2756369
2: -0.0024124, 0.2897211, 0.0167517, 0.2375837, -0.2399961, 0.2729694
3: -0.0441975, 0.1421833, -0.0251918, 0.0982481, -0.1424456, 0.1673751
4: -0.0039181, 0.2695846, 0.0184394, 0.2241397, -0.2280577, 0.2511452

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0359142, 0.2811165, 0.0238402, 0.0717446, -0.1076588, 0.2572763
1: -0.0888036, 0.2224801, -0.0526383, 0.2019492, -0.2907528, 0.2751183
2: -0.0024124, 0.2897211, 0.0178157, 0.2683529, -0.2707652, 0.2719054
3: -0.0441975, 0.1421833, -0.0227460, 0.1140152, -0.1582127, 0.1649293
4: -0.0039181, 0.2695846, 0.0202869, 0.2539533, -0.2578713, 0.2492977

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, 0.0179609, 0.0712747, -0.0844061, 0.1317447
1: -0.0775794, 0.2161378, -0.0531569, 0.1766041, -0.2541836, 0.2692947
2: 0.0068115, 0.2863497, 0.0167517, 0.2375837, -0.2307722, 0.2695980
3: -0.0347106, 0.1325780, -0.0251918, 0.0982481, -0.1329587, 0.1577698
4: 0.0104636, 0.2701671, 0.0184394, 0.2241397, -0.2136761, 0.2517277

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8668168
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8659791
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, 0.0238402, 0.0717446, -0.0848760, 0.1258654
1: -0.0775794, 0.2161378, -0.0526383, 0.2019492, -0.2795287, 0.2687761
2: 0.0068115, 0.2863497, 0.0178157, 0.2683529, -0.2615413, 0.2685340
3: -0.0347106, 0.1325780, -0.0227460, 0.1140152, -0.1487257, 0.1553240
4: 0.0104636, 0.2701671, 0.0202869, 0.2539533, -0.2434896, 0.2498803

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8668169
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8659791
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0359142, 0.2811165, -0.3174485, 0.3043096
1: -0.0901904, 0.2714047, -0.0888036, 0.2224801, -0.3126704, 0.3602083
2: -0.0009017, 0.3494731, -0.0024124, 0.2897211, -0.2906228, 0.3518855
3: -0.0441245, 0.1647791, -0.0441975, 0.1421833, -0.1863078, 0.2089766
4: -0.0009160, 0.3274141, -0.0039181, 0.2695846, -0.2705006, 0.3313321

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8442480
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8645174
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0131314, 0.1497056, -0.1860375, 0.2815269
1: -0.0901904, 0.2714047, -0.0775794, 0.2161378, -0.3063282, 0.3489841
2: -0.0009017, 0.3494731, 0.0068115, 0.2863497, -0.2872514, 0.3426616
3: -0.0441245, 0.1647791, -0.0347106, 0.1325780, -0.1767025, 0.1994897
4: -0.0009160, 0.3274141, 0.0104636, 0.2701671, -0.2710831, 0.3169505

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8443284
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650914, upper bound: 0.8645978
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0359142, 0.2811165, -0.2935548, 0.2199977
1: -0.0840206, 0.2807598, -0.0888036, 0.2224801, -0.3065007, 0.3695634
2: 0.0052519, 0.3607540, -0.0024124, 0.2897211, -0.2844691, 0.3631663
3: -0.0381413, 0.1672755, -0.0441975, 0.1421833, -0.1803246, 0.2114730
4: 0.0080186, 0.3421003, -0.0039181, 0.2695846, -0.2615660, 0.3460184

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8648318, upper bound: 0.8521999
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0131314, 0.1497056, -0.1621439, 0.1972150
1: -0.0840206, 0.2807598, -0.0775794, 0.2161378, -0.3001584, 0.3583393
2: 0.0052519, 0.3607540, 0.0068115, 0.2863497, -0.2810978, 0.3539424
3: -0.0381413, 0.1672755, -0.0347106, 0.1325780, -0.1707193, 0.2019861
4: 0.0080186, 0.3421003, 0.0104636, 0.2701671, -0.2621485, 0.3316367

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8648318, upper bound: 0.8522746
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0119383, 0.0596709, -0.0960028, 0.2564571
1: -0.0901904, 0.2714047, -0.0526713, 0.1203077, -0.2104981, 0.3240759
2: -0.0009017, 0.3494731, 0.0184351, 0.1715650, -0.1724667, 0.3310380
3: -0.0441245, 0.1647791, -0.0221550, 0.0700969, -0.1142214, 0.1869341
4: -0.0009160, 0.3274141, 0.0211372, 0.1609437, -0.1618596, 0.3062769

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556635, upper bound: 0.8443540
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8556635, upper bound: 0.8646081
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0297914, 0.0525318, -0.0888637, 0.2386040
1: -0.0901904, 0.2714047, -0.0384549, 0.0809017, -0.1710921, 0.3098596
2: -0.0009017, 0.3494731, 0.0246098, 0.1272665, -0.1281682, 0.3248633
3: -0.0441245, 0.1647791, -0.0198346, 0.0443621, -0.0884866, 0.1846137
4: -0.0009160, 0.3274141, 0.0246516, 0.1186544, -0.1195703, 0.3027624

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557456, upper bound: 0.8443540
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8557456, upper bound: 0.8646102
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0119383, 0.0596709, -0.0721092, 0.1721452
1: -0.0840206, 0.2807598, -0.0526713, 0.1203077, -0.2043283, 0.3334311
2: 0.0052519, 0.3607540, 0.0184351, 0.1715650, -0.1663130, 0.3423188
3: -0.0381413, 0.1672755, -0.0221550, 0.0700969, -0.1082382, 0.1894305
4: 0.0080186, 0.3421003, 0.0211372, 0.1609437, -0.1529251, 0.3209631

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8560744, upper bound: 0.8523059
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8560744, upper bound: 0.8622517
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0297914, 0.0525318, -0.0649701, 0.1542921
1: -0.0840206, 0.2807598, -0.0384549, 0.0809017, -0.1649223, 0.3192147
2: 0.0052519, 0.3607540, 0.0246098, 0.1272665, -0.1220145, 0.3361442
3: -0.0381413, 0.1672755, -0.0198346, 0.0443621, -0.0825034, 0.1871101
4: 0.0080186, 0.3421003, 0.0246516, 0.1186544, -0.1106358, 0.3174487

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8561565, upper bound: 0.8523059
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561565, upper bound: 0.8622517
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0359142, 0.2811165, -0.2631557, 0.1071889
1: -0.0531569, 0.1766041, -0.0888036, 0.2224801, -0.2756369, 0.2654077
2: 0.0167517, 0.2375837, -0.0024124, 0.2897211, -0.2729694, 0.2399961
3: -0.0251918, 0.0982481, -0.0441975, 0.1421833, -0.1673751, 0.1424456
4: 0.0184394, 0.2241397, -0.0039181, 0.2695846, -0.2511452, 0.2280577

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8648599, upper bound: 0.8516314
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8651444, upper bound: 0.8606791
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0131314, 0.1497056, -0.1317447, 0.0844061
1: -0.0531569, 0.1766041, -0.0775794, 0.2161378, -0.2692947, 0.2541836
2: 0.0167517, 0.2375837, 0.0068115, 0.2863497, -0.2695980, 0.2307722
3: -0.0251918, 0.0982481, -0.0347106, 0.1325780, -0.1577698, 0.1329587
4: 0.0184394, 0.2241397, 0.0104636, 0.2701671, -0.2517277, 0.2136761

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8648599, upper bound: 0.8517118
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8651444, upper bound: 0.8607595
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, 0.0119383, 0.0596709, -0.0417100, 0.0593364
1: -0.0531569, 0.1766041, -0.0526713, 0.1203077, -0.1734646, 0.2292754
2: 0.0167517, 0.2375837, 0.0184351, 0.1715650, -0.1548133, 0.2191486
3: -0.0251918, 0.0982481, -0.0221550, 0.0700969, -0.0952887, 0.1204031
4: 0.0184394, 0.2241397, 0.0211372, 0.1609437, -0.1425042, 0.2030025

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8565391, upper bound: 0.8517374
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8565391, upper bound: 0.8607461
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, 0.0297914, 0.0525318, -0.0345709, 0.0414833
1: -0.0531569, 0.1766041, -0.0384549, 0.0809017, -0.1340586, 0.2150590
2: 0.0167517, 0.2375837, 0.0246098, 0.1272665, -0.1105148, 0.2129739
3: -0.0251918, 0.0982481, -0.0198346, 0.0443621, -0.0695539, 0.1180827
4: 0.0184394, 0.2241397, 0.0246516, 0.1186544, -0.1002149, 0.1994881

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8517374
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8607461
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0363319, 0.2683954, -0.3047274, 0.3047274
1: -0.0901904, 0.2714047, -0.0901904, 0.2714047, -0.3615950, 0.3615950
2: -0.0009017, 0.3494731, -0.0009017, 0.3494731, -0.3503748, 0.3503748
3: -0.0441245, 0.1647791, -0.0441245, 0.1647791, -0.2089036, 0.2089036
4: -0.0009160, 0.3274141, -0.0009160, 0.3274141, -0.3283301, 0.3283301

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8617155, upper bound: 0.8442480
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8623859, upper bound: 0.8645798
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, -0.0124383, 0.1840835, -0.2204155, 0.2808337
1: -0.0901904, 0.2714047, -0.0840206, 0.2807598, -0.3709502, 0.3554253
2: -0.0009017, 0.3494731, 0.0052519, 0.3607540, -0.3616557, 0.3442212
3: -0.0441245, 0.1647791, -0.0381413, 0.1672755, -0.2114001, 0.2029204
4: -0.0009160, 0.3274141, 0.0080186, 0.3421003, -0.3430163, 0.3193955

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8617155, upper bound: 0.8443168
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8623859, upper bound: 0.8645861
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0363319, 0.2683954, -0.2808337, 0.2204155
1: -0.0840206, 0.2807598, -0.0901904, 0.2714047, -0.3554253, 0.3709502
2: 0.0052519, 0.3607540, -0.0009017, 0.3494731, -0.3442212, 0.3616557
3: -0.0381413, 0.1672755, -0.0441245, 0.1647791, -0.2029204, 0.2114001
4: 0.0080186, 0.3421003, -0.0009160, 0.3274141, -0.3193955, 0.3430163

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621264, upper bound: 0.8521999
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0124383, 0.1840835, -0.1965218, 0.1965218
1: -0.0840206, 0.2807598, -0.0840206, 0.2807598, -0.3647804, 0.3647804
2: 0.0052519, 0.3607540, 0.0052519, 0.3607540, -0.3555020, 0.3555020
3: -0.0381413, 0.1672755, -0.0381413, 0.1672755, -0.2054169, 0.2054169
4: 0.0080186, 0.3421003, 0.0080186, 0.3421003, -0.3340817, 0.3340817

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8621264, upper bound: 0.8522686
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0179609, 0.0712747, -0.1076066, 0.2504345
1: -0.0901904, 0.2714047, -0.0531569, 0.1766041, -0.2667945, 0.3245615
2: -0.0009017, 0.3494731, 0.0167517, 0.2375837, -0.2384854, 0.3327214
3: -0.0441245, 0.1647791, -0.0251918, 0.0982481, -0.1423726, 0.1899709
4: -0.0009160, 0.3274141, 0.0184394, 0.2241397, -0.2250556, 0.3089747

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8534110, upper bound: 0.8439850
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646263
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0363319, 0.2683954, 0.0238402, 0.0717446, -0.1080765, 0.2445552
1: -0.0901904, 0.2714047, -0.0526383, 0.2019492, -0.2921396, 0.3240429
2: -0.0009017, 0.3494731, 0.0178157, 0.2683529, -0.2692546, 0.3316574
3: -0.0441245, 0.1647791, -0.0227460, 0.1140152, -0.1581397, 0.1875251
4: -0.0009160, 0.3274141, 0.0202869, 0.2539533, -0.2548693, 0.3071272

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8534110, upper bound: 0.8443540
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646263
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0179609, 0.0712747, -0.0837130, 0.1661226
1: -0.0840206, 0.2807598, -0.0531569, 0.1766041, -0.2606247, 0.3339167
2: 0.0052519, 0.3607540, 0.0167517, 0.2375837, -0.2323318, 0.3440022
3: -0.0381413, 0.1672755, -0.0251918, 0.0982481, -0.1363894, 0.1924674
4: 0.0080186, 0.3421003, 0.0184394, 0.2241397, -0.2161211, 0.3236609

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8536503, upper bound: 0.8519368
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573417, upper bound: 0.8618827
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, 0.0238402, 0.0717446, -0.0841829, 0.1602433
1: -0.0840206, 0.2807598, -0.0526383, 0.2019492, -0.2859699, 0.3333981
2: 0.0052519, 0.3607540, 0.0178157, 0.2683529, -0.2631009, 0.3429383
3: -0.0381413, 0.1672755, -0.0227460, 0.1140152, -0.1521565, 0.1900215
4: 0.0080186, 0.3421003, 0.0202869, 0.2539533, -0.2459347, 0.3218135

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8536503, upper bound: 0.8523056
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573417, upper bound: 0.8618827
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0363319, 0.2683954, -0.2504345, 0.1076066
1: -0.0531569, 0.1766041, -0.0901904, 0.2714047, -0.3245615, 0.2667945
2: 0.0167517, 0.2375837, -0.0009017, 0.3494731, -0.3327214, 0.2384854
3: -0.0251918, 0.0982481, -0.0441245, 0.1647791, -0.1899709, 0.1423726
4: 0.0184394, 0.2241397, -0.0009160, 0.3274141, -0.3089747, 0.2250556

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8625911, upper bound: 0.8517029
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8627325, upper bound: 0.8607506
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0712747, -0.0124383, 0.1840835, -0.1661226, 0.0837130
1: -0.0531569, 0.1766041, -0.0840206, 0.2807598, -0.3339167, 0.2606247
2: 0.0167517, 0.2375837, 0.0052519, 0.3607540, -0.3440022, 0.2323318
3: -0.0251918, 0.0982481, -0.0381413, 0.1672755, -0.1924674, 0.1363894
4: 0.0184394, 0.2241397, 0.0080186, 0.3421003, -0.3236609, 0.2161211

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8625911, upper bound: 0.8517029
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8627325, upper bound: 0.8607506
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.19 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8462533
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8649227
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8617182, upper bound: 0.8462533
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8649227
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8621458, upper bound: 0.8658886
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8666686
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8622262, upper bound: 0.8658886
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8668168
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8659791
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8668169
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573515, upper bound: 0.8659791
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8442480
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8645174
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8644209, upper bound: 0.8443284
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8650914, upper bound: 0.8645978
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8648318, upper bound: 0.8521999
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8648318, upper bound: 0.8522746
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8649227, upper bound: 0.8621458
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8556635, upper bound: 0.8443540
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8556635, upper bound: 0.8646081
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8557456, upper bound: 0.8443540
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8557456, upper bound: 0.8646102
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8560744, upper bound: 0.8523059
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8560744, upper bound: 0.8622517
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8561565, upper bound: 0.8523059
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8561565, upper bound: 0.8622517
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8648599, upper bound: 0.8516314
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8651444, upper bound: 0.8606791
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8648599, upper bound: 0.8517118
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8651444, upper bound: 0.8607595
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8565391, upper bound: 0.8517374
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8565391, upper bound: 0.8607461
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8517374
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8607461
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8617155, upper bound: 0.8442480
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8623859, upper bound: 0.8645798
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8617155, upper bound: 0.8443168
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8623859, upper bound: 0.8645861
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8621264, upper bound: 0.8521999
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8621264, upper bound: 0.8522686
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8622173, upper bound: 0.8621458
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8534110, upper bound: 0.8439850
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646263
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8534110, upper bound: 0.8443540
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646263
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8536503, upper bound: 0.8519368
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573417, upper bound: 0.8618827
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8536503, upper bound: 0.8523056
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8573417, upper bound: 0.8618827
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8625911, upper bound: 0.8517029
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8627325, upper bound: 0.8607506
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8625911, upper bound: 0.8517029
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -0.8627325, upper bound: 0.8607506

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0410866, 0.3006844, -0.0363319, 0.2683954, -0.3094820, 0.3370163
1: -0.0880179, 0.2315971, -0.0901904, 0.2714047, -0.3594226, 0.3217875
2: -0.0017034, 0.3005701, -0.0009017, 0.3494731, -0.3511765, 0.3014718
3: -0.0441768, 0.1478275, -0.0441245, 0.1647791, -0.2089560, 0.1919520
4: -0.0035206, 0.2797467, -0.0009160, 0.3274141, -0.3309347, 0.2806626

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8457515
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8464220
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0363319, 0.2683954, -0.2919485, 0.2375686
1: -0.0760552, 0.2038522, -0.0901904, 0.2714047, -0.3474599, 0.2940425
2: 0.0042095, 0.2710823, -0.0009017, 0.3494731, -0.3452636, 0.2719840
3: -0.0383188, 0.1244072, -0.0441245, 0.1647791, -0.2030979, 0.1685317
4: 0.0045975, 0.2538497, -0.0009160, 0.3274141, -0.3228166, 0.2547657

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8644209
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8650914
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0410866, 0.3006844, -0.0124383, 0.1840835, -0.2251701, 0.3131227
1: -0.0880179, 0.2315971, -0.0840206, 0.2807598, -0.3687778, 0.3156177
2: -0.0017034, 0.3005701, 0.0052519, 0.3607540, -0.3624573, 0.2953181
3: -0.0441768, 0.1478275, -0.0381413, 0.1672755, -0.2114524, 0.1859688
4: -0.0035206, 0.2797467, 0.0080186, 0.3421003, -0.3456209, 0.2717281

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8517723, upper bound: 0.8461624
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8517723, upper bound: 0.8462533
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0124383, 0.1840835, -0.2076366, 0.2136749
1: -0.0760552, 0.2038522, -0.0840206, 0.2807598, -0.3568150, 0.2878728
2: 0.0042095, 0.2710823, 0.0052519, 0.3607540, -0.3565444, 0.2658303
3: -0.0383188, 0.1244072, -0.0381413, 0.1672755, -0.2055943, 0.1625485
4: 0.0045975, 0.2538497, 0.0080186, 0.3421003, -0.3375028, 0.2458311

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649227
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0363319, 0.2683954, -0.2973483, 0.2396285
1: -0.0891756, 0.2573025, -0.0901904, 0.2714047, -0.3605803, 0.3474929
2: 0.0028422, 0.3348821, -0.0009017, 0.3494731, -0.3466309, 0.3357838
3: -0.0384304, 0.1591823, -0.0441245, 0.1647791, -0.2032095, 0.2033068
4: 0.0068889, 0.3157227, -0.0009160, 0.3274141, -0.3205252, 0.3166387

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8653868
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8660573
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0363319, 0.2683954, -0.2715902, 0.1361581
1: -0.0681015, 0.2029338, -0.0901904, 0.2714047, -0.3395061, 0.2931241
2: 0.0107180, 0.2719691, -0.0009017, 0.3494731, -0.3387551, 0.2728708
3: -0.0302884, 0.1220847, -0.0441245, 0.1647791, -0.1950676, 0.1662092
4: 0.0149963, 0.2578355, -0.0009160, 0.3274141, -0.3124178, 0.2587515

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8653868
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8660573
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0124383, 0.1840835, -0.2130364, 0.2157349
1: -0.0891756, 0.2573025, -0.0840206, 0.2807598, -0.3699355, 0.3413231
2: 0.0028422, 0.3348821, 0.0052519, 0.3607540, -0.3579118, 0.3296302
3: -0.0384304, 0.1591823, -0.0381413, 0.1672755, -0.2057060, 0.1973236
4: 0.0068889, 0.3157227, 0.0080186, 0.3421003, -0.3352114, 0.3077041

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8657977
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8522803, upper bound: 0.8658886
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0124383, 0.1840835, -0.1872783, 0.1122644
1: -0.0681015, 0.2029338, -0.0840206, 0.2807598, -0.3488613, 0.2869544
2: 0.0107180, 0.2719691, 0.0052519, 0.3607540, -0.3500359, 0.2667172
3: -0.0302884, 0.1220847, -0.0381413, 0.1672755, -0.1975640, 0.1602260
4: 0.0149963, 0.2578355, 0.0080186, 0.3421003, -0.3271040, 0.2498169

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8657977
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8522803, upper bound: 0.8658886
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0179609, 0.0712747, -0.0948278, 0.1832758
1: -0.0760552, 0.2038522, -0.0531569, 0.1766041, -0.2526594, 0.2570090
2: 0.0042095, 0.2710823, 0.0167517, 0.2375837, -0.2333742, 0.2543306
3: -0.0383188, 0.1244072, -0.0251918, 0.0982481, -0.1365669, 0.1495990
4: 0.0045975, 0.2538497, 0.0184394, 0.2241397, -0.2195421, 0.2354103

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8516314, upper bound: 0.8648599
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8516314, upper bound: 0.8651444
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0238402, 0.0717446, -0.0952976, 0.1773964
1: -0.0760552, 0.2038522, -0.0526383, 0.2019492, -0.2780045, 0.2564904
2: 0.0042095, 0.2710823, 0.0178157, 0.2683529, -0.2641433, 0.2532666
3: -0.0383188, 0.1244072, -0.0227460, 0.1140152, -0.1523340, 0.1471532
4: 0.0045975, 0.2538497, 0.0202869, 0.2539533, -0.2493557, 0.2335629

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649663
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8650098
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, 0.0179609, 0.0712747, -0.1002275, 0.1853357
1: -0.0891756, 0.2573025, -0.0531569, 0.1766041, -0.2657798, 0.3104594
2: 0.0028422, 0.3348821, 0.0167517, 0.2375837, -0.2347415, 0.3181304
3: -0.0384304, 0.1591823, -0.0251918, 0.0982481, -0.1366785, 0.1843741
4: 0.0068889, 0.3157227, 0.0184394, 0.2241397, -0.2172508, 0.2972833

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8657460
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8660967
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, 0.0179609, 0.0712747, -0.0744694, 0.0818653
1: -0.0681015, 0.2029338, -0.0531569, 0.1766041, -0.2447056, 0.2560906
2: 0.0107180, 0.2719691, 0.0167517, 0.2375837, -0.2268657, 0.2552174
3: -0.0302884, 0.1220847, -0.0251918, 0.0982481, -0.1285366, 0.1472765
4: 0.0149963, 0.2578355, 0.0184394, 0.2241397, -0.2091434, 0.2393961

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8657460
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8660967
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, 0.0238402, 0.0717446, -0.1006974, 0.1794564
1: -0.0891756, 0.2573025, -0.0526383, 0.2019492, -0.2911249, 0.3099408
2: 0.0028422, 0.3348821, 0.0178157, 0.2683529, -0.2655107, 0.3170665
3: -0.0384304, 0.1591823, -0.0227460, 0.1140152, -0.1524456, 0.1819283
4: 0.0068889, 0.3157227, 0.0202869, 0.2539533, -0.2470644, 0.2954358

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659751
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659791
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, 0.0238402, 0.0717446, -0.0749393, 0.0759859
1: -0.0681015, 0.2029338, -0.0526383, 0.2019492, -0.2700507, 0.2555721
2: 0.0107180, 0.2719691, 0.0178157, 0.2683529, -0.2576348, 0.2541535
3: -0.0302884, 0.1220847, -0.0227460, 0.1140152, -0.1443036, 0.1448306
4: 0.0149963, 0.2578355, 0.0202869, 0.2539533, -0.2389570, 0.2375486

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659751
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659791
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0464171, 0.3033874, -0.0359142, 0.2811165, -0.3275336, 0.3393016
1: -0.0908616, 0.2861186, -0.0888036, 0.2224801, -0.3133416, 0.3749222
2: -0.0010029, 0.3666863, -0.0024124, 0.2897211, -0.2907240, 0.3690987
3: -0.0452064, 0.1740732, -0.0441975, 0.1421833, -0.1873897, 0.2182707
4: -0.0018012, 0.3433452, -0.0039181, 0.2695846, -0.2713858, 0.3472632

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, -0.0359142, 0.2811165, -0.3054404, 0.2304641
1: -0.0778679, 0.2563584, -0.0888036, 0.2224801, -0.3003479, 0.3451620
2: 0.0041553, 0.3334116, -0.0024124, 0.2897211, -0.2855657, 0.3358240
3: -0.0386840, 0.1512785, -0.0441975, 0.1421833, -0.1808673, 0.1954760
4: 0.0057370, 0.3140403, -0.0039181, 0.2695846, -0.2638476, 0.3179584

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8464220, upper bound: 0.8640898
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8464220, upper bound: 0.8645174
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0464171, 0.3033874, -0.0131314, 0.1497056, -0.1961227, 0.3165188
1: -0.0908616, 0.2861186, -0.0775794, 0.2161378, -0.3069994, 0.3636980
2: -0.0010029, 0.3666863, 0.0068115, 0.2863497, -0.2873527, 0.3598748
3: -0.0452064, 0.1740732, -0.0347106, 0.1325780, -0.1777844, 0.2087837
4: -0.0018012, 0.3433452, 0.0104636, 0.2701671, -0.2719683, 0.3328815

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653868, upper bound: 0.8443284
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653868, upper bound: 0.8443284
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, -0.0131314, 0.1497056, -0.1740294, 0.2076813
1: -0.0778679, 0.2563584, -0.0775794, 0.2161378, -0.2940057, 0.3339378
2: 0.0041553, 0.3334116, 0.0068115, 0.2863497, -0.2821944, 0.3266000
3: -0.0386840, 0.1512785, -0.0347106, 0.1325780, -0.1712619, 0.1859891
4: 0.0057370, 0.3140403, 0.0104636, 0.2701671, -0.2644301, 0.3035767

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8660573, upper bound: 0.8645978
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8660573, upper bound: 0.8645978
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0349724, 0.2740249, -0.0359142, 0.2811165, -0.3160889, 0.3099391
1: -0.0981120, 0.3256595, -0.0888036, 0.2224801, -0.3205920, 0.4144631
2: 0.0005621, 0.4139100, -0.0024124, 0.2897211, -0.2891590, 0.4163224
3: -0.0445257, 0.1959200, -0.0441975, 0.1421833, -0.1867090, 0.2401175
4: 0.0021093, 0.3914595, -0.0039181, 0.2695846, -0.2674753, 0.3953775

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8517723
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, -0.0359142, 0.2811165, -0.2841718, 0.1885260
1: -0.0757312, 0.2676867, -0.0888036, 0.2224801, -0.2982112, 0.3564903
2: 0.0086105, 0.3466829, -0.0024124, 0.2897211, -0.2811105, 0.3490953
3: -0.0342591, 0.1571698, -0.0441975, 0.1421833, -0.1764424, 0.2013673
4: 0.0115876, 0.3295513, -0.0039181, 0.2695846, -0.2579970, 0.3334694

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8462533, upper bound: 0.8617182
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8462533, upper bound: 0.8621458
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0349724, 0.2740249, -0.0131314, 0.1497056, -0.1846780, 0.2871563
1: -0.0981120, 0.3256595, -0.0775794, 0.2161378, -0.3142498, 0.4032390
2: 0.0005621, 0.4139100, 0.0068115, 0.2863497, -0.2857876, 0.4070985
3: -0.0445257, 0.1959200, -0.0347106, 0.1325780, -0.1771037, 0.2306305
4: 0.0021093, 0.3914595, 0.0104636, 0.2701671, -0.2680579, 0.3809958

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8521826
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8522746
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, -0.0131314, 0.1497056, -0.1527609, 0.1657433
1: -0.0757312, 0.2676867, -0.0775794, 0.2161378, -0.2918690, 0.3452662
2: 0.0086105, 0.3466829, 0.0068115, 0.2863497, -0.2777392, 0.3398714
3: -0.0342591, 0.1571698, -0.0347106, 0.1325780, -0.1668371, 0.1918804
4: 0.0115876, 0.3295513, 0.0104636, 0.2701671, -0.2585795, 0.3190877

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8617182
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8621458
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, 0.0119383, 0.0596709, -0.0839948, 0.1826116
1: -0.0778679, 0.2563584, -0.0526713, 0.1203077, -0.1981756, 0.3090296
2: 0.0041553, 0.3334116, 0.0184351, 0.1715650, -0.1674096, 0.3149765
3: -0.0386840, 0.1512785, -0.0221550, 0.0700969, -0.1087808, 0.1734335
4: 0.0057370, 0.3140403, 0.0211372, 0.1609437, -0.1552066, 0.2929031

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8641877
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8646081
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, 0.0297914, 0.0525318, -0.0768556, 0.1647584
1: -0.0778679, 0.2563584, -0.0384549, 0.0809017, -0.1587696, 0.2948133
2: 0.0041553, 0.3334116, 0.0246098, 0.1272665, -0.1231111, 0.3088018
3: -0.0386840, 0.1512785, -0.0198346, 0.0443621, -0.0830461, 0.1711131
4: 0.0057370, 0.3140403, 0.0246516, 0.1186544, -0.1129173, 0.2893887

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8642543
time: 0.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8646102
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, 0.0119383, 0.0596709, -0.0627262, 0.1406735
1: -0.0757312, 0.2676867, -0.0526713, 0.1203077, -0.1960389, 0.3203580
2: 0.0086105, 0.3466829, 0.0184351, 0.1715650, -0.1629544, 0.3282478
3: -0.0342591, 0.1571698, -0.0221550, 0.0700969, -0.1043560, 0.1793248
4: 0.0115876, 0.3295513, 0.0211372, 0.1609437, -0.1493561, 0.3084141

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618310
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618310
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, 0.0297914, 0.0525318, -0.0555870, 0.1228204
1: -0.0757312, 0.2676867, -0.0384549, 0.0809017, -0.1566329, 0.3061416
2: 0.0086105, 0.3466829, 0.0246098, 0.1272665, -0.1186559, 0.3220731
3: -0.0342591, 0.1571698, -0.0198346, 0.0443621, -0.0786213, 0.1770044
4: 0.0115876, 0.3295513, 0.0246516, 0.1186544, -0.1070668, 0.3048997

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618827
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618827
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0152065, 0.0719816, -0.0359142, 0.2811165, -0.2659100, 0.1078958
1: -0.0526925, 0.1818441, -0.0888036, 0.2224801, -0.2751725, 0.2706477
2: 0.0169597, 0.2437519, -0.0024124, 0.2897211, -0.2727613, 0.2461643
3: -0.0247090, 0.1013449, -0.0441975, 0.1421833, -0.1668923, 0.1455424
4: 0.0185986, 0.2299674, -0.0039181, 0.2695846, -0.2509860, 0.2338854

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8466271, upper bound: 0.8512038
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8466271, upper bound: 0.8516314
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0359142, 0.2811165, -0.2544437, 0.0993636
1: -0.0436111, 0.1676113, -0.0888036, 0.2224801, -0.2660911, 0.2564149
2: 0.0208580, 0.2285865, -0.0024124, 0.2897211, -0.2688631, 0.2309989
3: -0.0223003, 0.0900128, -0.0441975, 0.1421833, -0.1644836, 0.1342103
4: 0.0212943, 0.2158635, -0.0039181, 0.2695846, -0.2482902, 0.2197816

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467685, upper bound: 0.8602515
time: 0.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8467685, upper bound: 0.8606791
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0152065, 0.0719816, -0.0131314, 0.1497056, -0.1344991, 0.0851130
1: -0.0526925, 0.1818441, -0.0775794, 0.2161378, -0.2688303, 0.2594236
2: 0.0169597, 0.2437519, 0.0068115, 0.2863497, -0.2693900, 0.2369404
3: -0.0247090, 0.1013449, -0.0347106, 0.1325780, -0.1572870, 0.1360555
4: 0.0185986, 0.2299674, 0.0104636, 0.2701671, -0.2515685, 0.2195037

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657460, upper bound: 0.8517118
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657460, upper bound: 0.8517118
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0131314, 0.1497056, -0.1230327, 0.0765808
1: -0.0436111, 0.1676113, -0.0775794, 0.2161378, -0.2597489, 0.2451907
2: 0.0208580, 0.2285865, 0.0068115, 0.2863497, -0.2654917, 0.2217750
3: -0.0223003, 0.0900128, -0.0347106, 0.1325780, -0.1548783, 0.1247234
4: 0.0212943, 0.2158635, 0.0104636, 0.2701671, -0.2488728, 0.2053999

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8660967, upper bound: 0.8607595
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8660967, upper bound: 0.8607595
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0119383, 0.0596709, -0.0329980, 0.0515111
1: -0.0436111, 0.1676113, -0.0526713, 0.1203077, -0.1639188, 0.2202826
2: 0.0208580, 0.2285865, 0.0184351, 0.1715650, -0.1507070, 0.2101514
3: -0.0223003, 0.0900128, -0.0221550, 0.0700969, -0.0923972, 0.1121678
4: 0.0212943, 0.2158635, 0.0211372, 0.1609437, -0.1396493, 0.1947263

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8602515
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8607461
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0297914, 0.0525318, -0.0258589, 0.0336579
1: -0.0436111, 0.1676113, -0.0384549, 0.0809017, -0.1245128, 0.2060662
2: 0.0208580, 0.2285865, 0.0246098, 0.1272665, -0.1064085, 0.2039767
3: -0.0223003, 0.0900128, -0.0198346, 0.0443621, -0.0666624, 0.1098474
4: 0.0212943, 0.2158635, 0.0246516, 0.1186544, -0.0973600, 0.1912119

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8604160
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8607461
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0464171, 0.3033874, -0.0363319, 0.2683954, -0.3148125, 0.3397193
1: -0.0908616, 0.2861186, -0.0901904, 0.2714047, -0.3622662, 0.3763089
2: -0.0010029, 0.3666863, -0.0009017, 0.3494731, -0.3504761, 0.3675880
3: -0.0452064, 0.1740732, -0.0441245, 0.1647791, -0.2099856, 0.2181977
4: -0.0018012, 0.3433452, -0.0009160, 0.3274141, -0.3292153, 0.3442611

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, -0.0363319, 0.2683954, -0.2927193, 0.2308818
1: -0.0778679, 0.2563584, -0.0901904, 0.2714047, -0.3492725, 0.3465487
2: 0.0041553, 0.3334116, -0.0009017, 0.3494731, -0.3453178, 0.3343133
3: -0.0386840, 0.1512785, -0.0441245, 0.1647791, -0.2034631, 0.1954030
4: 0.0057370, 0.3140403, -0.0009160, 0.3274141, -0.3216771, 0.3149563

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8444882, upper bound: 0.8640871
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8645819
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0464171, 0.3033874, -0.0124383, 0.1840835, -0.2305006, 0.3158257
1: -0.0908616, 0.2861186, -0.0840206, 0.2807598, -0.3716214, 0.3701392
2: -0.0010029, 0.3666863, 0.0052519, 0.3607540, -0.3617569, 0.3614343
3: -0.0452064, 0.1740732, -0.0381413, 0.1672755, -0.2124820, 0.2122145
4: -0.0018012, 0.3433452, 0.0080186, 0.3421003, -0.3439015, 0.3353266

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8517696, upper bound: 0.8442286
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8517696, upper bound: 0.8443168
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, -0.0124383, 0.1840835, -0.2084074, 0.2069882
1: -0.0778679, 0.2563584, -0.0840206, 0.2807598, -0.3586277, 0.3403790
2: 0.0041553, 0.3334116, 0.0052519, 0.3607540, -0.3565986, 0.3281596
3: -0.0386840, 0.1512785, -0.0381413, 0.1672755, -0.2059595, 0.1894198
4: 0.0057370, 0.3140403, 0.0080186, 0.3421003, -0.3363633, 0.3060217

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8644980
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8645861
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0349724, 0.2740249, -0.0363319, 0.2683954, -0.3033678, 0.3103568
1: -0.0981120, 0.3256595, -0.0901904, 0.2714047, -0.3695167, 0.4158499
2: 0.0005621, 0.4139100, -0.0009017, 0.3494731, -0.3489110, 0.4148117
3: -0.0445257, 0.1959200, -0.0441245, 0.1647791, -0.2093048, 0.2400445
4: 0.0021093, 0.3914595, -0.0009160, 0.3274141, -0.3253048, 0.3923754

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8442286, upper bound: 0.8517696
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8442286, upper bound: 0.8521999
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, -0.0363319, 0.2683954, -0.2714507, 0.1889438
1: -0.0757312, 0.2676867, -0.0901904, 0.2714047, -0.3471359, 0.3578771
2: 0.0086105, 0.3466829, -0.0009017, 0.3494731, -0.3408626, 0.3475846
3: -0.0342591, 0.1571698, -0.0441245, 0.1647791, -0.1990383, 0.2012943
4: 0.0115876, 0.3295513, -0.0009160, 0.3274141, -0.3158265, 0.3304673

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443195, upper bound: 0.8617155
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8443168, upper bound: 0.8621458
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0349724, 0.2740249, -0.0124383, 0.1840835, -0.2190559, 0.2864632
1: -0.0981120, 0.3256595, -0.0840206, 0.2807598, -0.3788718, 0.4096801
2: 0.0005621, 0.4139100, 0.0052519, 0.3607540, -0.3601919, 0.4086581
3: -0.0445257, 0.1959200, -0.0381413, 0.1672755, -0.2118012, 0.2340613
4: 0.0021093, 0.3914595, 0.0080186, 0.3421003, -0.3399911, 0.3834409

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8521805
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8522686
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, -0.0124383, 0.1840835, -0.1871388, 0.1650501
1: -0.0757312, 0.2676867, -0.0840206, 0.2807598, -0.3564910, 0.3517073
2: 0.0086105, 0.3466829, 0.0052519, 0.3607540, -0.3521434, 0.3414310
3: -0.0342591, 0.1571698, -0.0381413, 0.1672755, -0.2015347, 0.1953111
4: 0.0115876, 0.3295513, 0.0080186, 0.3421003, -0.3305127, 0.3215328

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8617155
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8621458
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, 0.0179609, 0.0712747, -0.0955985, 0.1765890
1: -0.0778679, 0.2563584, -0.0531569, 0.1766041, -0.2544720, 0.3095152
2: 0.0041553, 0.3334116, 0.0167517, 0.2375837, -0.2334284, 0.3166599
3: -0.0386840, 0.1512785, -0.0251918, 0.0982481, -0.1369321, 0.1764703
4: 0.0057370, 0.3140403, 0.0184394, 0.2241397, -0.2184027, 0.2956009

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8518716, upper bound: 0.8644691
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8518716, upper bound: 0.8646906
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0243238, 0.1945499, 0.0238402, 0.0717446, -0.0960684, 0.1707097
1: -0.0778679, 0.2563584, -0.0526383, 0.2019492, -0.2798171, 0.3089966
2: 0.0041553, 0.3334116, 0.0178157, 0.2683529, -0.2641975, 0.3155959
3: -0.0386840, 0.1512785, -0.0227460, 0.1140152, -0.1526991, 0.1740245
4: 0.0057370, 0.3140403, 0.0202869, 0.2539533, -0.2482162, 0.2937534

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646109
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646263
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, 0.0179609, 0.0712747, -0.0743300, 0.1346510
1: -0.0757312, 0.2676867, -0.0531569, 0.1766041, -0.2523353, 0.3208436
2: 0.0086105, 0.3466829, 0.0167517, 0.2375837, -0.2289732, 0.3299312
3: -0.0342591, 0.1571698, -0.0251918, 0.0982481, -0.1325073, 0.1823616
4: 0.0115876, 0.3295513, 0.0184394, 0.2241397, -0.2125521, 0.3111119

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8499045, upper bound: 0.8618310
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8499044, upper bound: 0.8618827
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, 0.0238402, 0.0717446, -0.0747998, 0.1287716
1: -0.0757312, 0.2676867, -0.0526383, 0.2019492, -0.2776804, 0.3203250
2: 0.0086105, 0.3466829, 0.0178157, 0.2683529, -0.2597423, 0.3288672
3: -0.0342591, 0.1571698, -0.0227460, 0.1140152, -0.1482743, 0.1799158
4: 0.0115876, 0.3295513, 0.0202869, 0.2539533, -0.2423657, 0.3092645

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547521, upper bound: 0.8618310
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547522, upper bound: 0.8618827
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0152065, 0.0719816, -0.0363319, 0.2683954, -0.2531889, 0.1083135
1: -0.0526925, 0.1818441, -0.0901904, 0.2714047, -0.3240972, 0.2720345
2: 0.0169597, 0.2437519, -0.0009017, 0.3494731, -0.3325134, 0.2446536
3: -0.0247090, 0.1013449, -0.0441245, 0.1647791, -0.1894882, 0.1454694
4: 0.0185986, 0.2299674, -0.0009160, 0.3274141, -0.3088155, 0.2308833

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8446933, upper bound: 0.8512011
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8446933, upper bound: 0.8517401
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0363319, 0.2683954, -0.2417225, 0.0997813
1: -0.0436111, 0.1676113, -0.0901904, 0.2714047, -0.3150158, 0.2578017
2: 0.0208580, 0.2285865, -0.0009017, 0.3494731, -0.3286151, 0.2294882
3: -0.0223003, 0.0900128, -0.0441245, 0.1647791, -0.1870794, 0.1341373
4: 0.0212943, 0.2158635, -0.0009160, 0.3274141, -0.3061197, 0.2167795

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8448347, upper bound: 0.8602488
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8448347, upper bound: 0.8609193
time: 0.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0152065, 0.0719816, -0.0124383, 0.1840835, -0.1688770, 0.0844199
1: -0.0526925, 0.1818441, -0.0840206, 0.2807598, -0.3334523, 0.2658648
2: 0.0169597, 0.2437519, 0.0052519, 0.3607540, -0.3437942, 0.2385000
3: -0.0247090, 0.1013449, -0.0381413, 0.1672755, -0.1919846, 0.1394862
4: 0.0185986, 0.2299674, 0.0080186, 0.3421003, -0.3235017, 0.2219488

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8526452, upper bound: 0.8516120
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8526452, upper bound: 0.8517029
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0124383, 0.1840835, -0.1574106, 0.0758877
1: -0.0436111, 0.1676113, -0.0840206, 0.2807598, -0.3243709, 0.2516319
2: 0.0208580, 0.2285865, 0.0052519, 0.3607540, -0.3398960, 0.2233346
3: -0.0223003, 0.0900128, -0.0381413, 0.1672755, -0.1895759, 0.1281541
4: 0.0212943, 0.2158635, 0.0080186, 0.3421003, -0.3208060, 0.2078449

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607506
time: 0.36 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.41 seconds
NS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8457515
NS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8464220
NS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8644209
NS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8650914
NS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517723, upper bound: 0.8461624
NS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517723, upper bound: 0.8462533
NS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
NS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649227
NS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8653868
NS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8660573
NS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8653868
NS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443284, upper bound: 0.8660573
NS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8657977
NS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8522803, upper bound: 0.8658886
NS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8657977
NS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8522803, upper bound: 0.8658886
NS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8516314, upper bound: 0.8648599
NS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8516314, upper bound: 0.8651444
NS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649663
NS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8650098
NS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8657460
NS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8660967
NS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8657460
NS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517118, upper bound: 0.8660967
NS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659751
NS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659791
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659751
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8548590, upper bound: 0.8659791
NS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
NS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
NS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8464220, upper bound: 0.8640898
NS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8464220, upper bound: 0.8645174
NS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8653868, upper bound: 0.8443284
NS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8653868, upper bound: 0.8443284
NS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8660573, upper bound: 0.8645978
NS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8660573, upper bound: 0.8645978
NS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8517723
NS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
NS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8462533, upper bound: 0.8617182
NS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8462533, upper bound: 0.8621458
NS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8521826
NS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8522746
NS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8617182
NS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8542018, upper bound: 0.8621458
NS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8641877
NS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8646081
NS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8642543
NS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8463276, upper bound: 0.8646102
NS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618310
NS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618310
NS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618827
NS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467385, upper bound: 0.8618827
NS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8466271, upper bound: 0.8512038
NS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8466271, upper bound: 0.8516314
NS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467685, upper bound: 0.8602515
NS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8467685, upper bound: 0.8606791
NS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8657460, upper bound: 0.8517118
NS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8657460, upper bound: 0.8517118
NS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8660967, upper bound: 0.8607595
NS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8660967, upper bound: 0.8607595
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8602515
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8607461
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8604160
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8472032, upper bound: 0.8607461
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8444882, upper bound: 0.8640871
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442480, upper bound: 0.8645819
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517696, upper bound: 0.8442286
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8517696, upper bound: 0.8443168
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8644980
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8645861
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442286, upper bound: 0.8517696
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8442286, upper bound: 0.8521999
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443195, upper bound: 0.8617155
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8443168, upper bound: 0.8621458
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8521805
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8522686
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8617155
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8521805, upper bound: 0.8621458
NS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8518716, upper bound: 0.8644691
NS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8518716, upper bound: 0.8646906
NS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646109
NS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646263
NS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8499045, upper bound: 0.8618310
NS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8499044, upper bound: 0.8618827
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8547521, upper bound: 0.8618310
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8547522, upper bound: 0.8618827
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8446933, upper bound: 0.8512011
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8446933, upper bound: 0.8517401
NS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8448347, upper bound: 0.8602488
NS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8448347, upper bound: 0.8609193
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8526452, upper bound: 0.8516120
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8526452, upper bound: 0.8517029
NS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
NS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.41
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607506

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0464171, 0.3033874, -0.3269405, 0.2476537
1: -0.0760552, 0.2038522, -0.0908616, 0.2861186, -0.3621738, 0.2947137
2: 0.0042095, 0.2710823, -0.0010029, 0.3666863, -0.3624768, 0.2720852
3: -0.0383188, 0.1244072, -0.0452064, 0.1740732, -0.2123920, 0.1696136
4: 0.0045975, 0.2538497, -0.0018012, 0.3433452, -0.3387476, 0.2556509

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0243238, 0.1945499, -0.2181029, 0.2255605
1: -0.0760552, 0.2038522, -0.0778679, 0.2563584, -0.3324136, 0.2817200
2: 0.0042095, 0.2710823, 0.0041553, 0.3334116, -0.3292021, 0.2669269
3: -0.0383188, 0.1244072, -0.0386840, 0.1512785, -0.1895973, 0.1630912
4: 0.0045975, 0.2538497, 0.0057370, 0.3140403, -0.3094428, 0.2481127

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0349724, 0.2740249, -0.2975780, 0.2362090
1: -0.0760552, 0.2038522, -0.0981120, 0.3256595, -0.4017147, 0.3019642
2: 0.0042095, 0.2710823, 0.0005621, 0.4139100, -0.4097005, 0.2705202
3: -0.0383188, 0.1244072, -0.0445257, 0.1959200, -0.2342388, 0.1689329
4: 0.0045975, 0.2538497, 0.0021093, 0.3914595, -0.3868619, 0.2517405

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, -0.0030553, 0.1526119, -0.1761649, 0.2042919
1: -0.0760552, 0.2038522, -0.0757312, 0.2676867, -0.3437420, 0.2795834
2: 0.0042095, 0.2710823, 0.0086105, 0.3466829, -0.3424734, 0.2624717
3: -0.0383188, 0.1244072, -0.0342591, 0.1571698, -0.1954886, 0.1586663
4: 0.0045975, 0.2538497, 0.0115876, 0.3295513, -0.3249538, 0.2422622

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0464171, 0.3033874, -0.3323402, 0.2497137
1: -0.0891756, 0.2573025, -0.0908616, 0.2861186, -0.3752942, 0.3481641
2: 0.0028422, 0.3348821, -0.0010029, 0.3666863, -0.3638441, 0.3358851
3: -0.0384304, 0.1591823, -0.0452064, 0.1740732, -0.2125036, 0.2043888
4: 0.0068889, 0.3157227, -0.0018012, 0.3433452, -0.3364563, 0.3175239

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0243238, 0.1945499, -0.2235027, 0.2276204
1: -0.0891756, 0.2573025, -0.0778679, 0.2563584, -0.3455340, 0.3351704
2: 0.0028422, 0.3348821, 0.0041553, 0.3334116, -0.3305694, 0.3307268
3: -0.0384304, 0.1591823, -0.0386840, 0.1512785, -0.1897089, 0.1978663
4: 0.0068889, 0.3157227, 0.0057370, 0.3140403, -0.3071514, 0.3099857

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0464171, 0.3033874, -0.3065821, 0.1462432
1: -0.0681015, 0.2029338, -0.0908616, 0.2861186, -0.3542200, 0.2937953
2: 0.0107180, 0.2719691, -0.0010029, 0.3666863, -0.3559683, 0.2729721
3: -0.0302884, 0.1220847, -0.0452064, 0.1740732, -0.2043616, 0.1672911
4: 0.0149963, 0.2578355, -0.0018012, 0.3433452, -0.3283488, 0.2596367

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0243238, 0.1945499, -0.1977446, 0.1241500
1: -0.0681015, 0.2029338, -0.0778679, 0.2563584, -0.3244599, 0.2808017
2: 0.0107180, 0.2719691, 0.0041553, 0.3334116, -0.3226936, 0.2678138
3: -0.0302884, 0.1220847, -0.0386840, 0.1512785, -0.1815669, 0.1607686
4: 0.0149963, 0.2578355, 0.0057370, 0.3140403, -0.2990440, 0.2520985

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0349724, 0.2740249, -0.3029777, 0.2382689
1: -0.0891756, 0.2573025, -0.0981120, 0.3256595, -0.4148352, 0.3554145
2: 0.0028422, 0.3348821, 0.0005621, 0.4139100, -0.4110678, 0.3343201
3: -0.0384304, 0.1591823, -0.0445257, 0.1959200, -0.2343504, 0.2037080
4: 0.0068889, 0.3157227, 0.0021093, 0.3914595, -0.3845706, 0.3136134

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, -0.0030553, 0.1526119, -0.1815647, 0.2063518
1: -0.0891756, 0.2573025, -0.0757312, 0.2676867, -0.3568624, 0.3330337
2: 0.0028422, 0.3348821, 0.0086105, 0.3466829, -0.3438407, 0.3262716
3: -0.0384304, 0.1591823, -0.0342591, 0.1571698, -0.1956002, 0.1934415
4: 0.0068889, 0.3157227, 0.0115876, 0.3295513, -0.3226624, 0.3041351

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0349724, 0.2740249, -0.2772196, 0.1347985
1: -0.0681015, 0.2029338, -0.0981120, 0.3256595, -0.3937610, 0.3010458
2: 0.0107180, 0.2719691, 0.0005621, 0.4139100, -0.4031920, 0.2714071
3: -0.0302884, 0.1220847, -0.0445257, 0.1959200, -0.2262084, 0.1666104
4: 0.0149963, 0.2578355, 0.0021093, 0.3914595, -0.3764631, 0.2557262

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0030553, 0.1526119, -0.1558066, 0.1028814
1: -0.0681015, 0.2029338, -0.0757312, 0.2676867, -0.3357882, 0.2786650
2: 0.0107180, 0.2719691, 0.0086105, 0.3466829, -0.3359649, 0.2633586
3: -0.0302884, 0.1220847, -0.0342591, 0.1571698, -0.1874582, 0.1563438
4: 0.0149963, 0.2578355, 0.0115876, 0.3295513, -0.3145550, 0.2462479

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0152065, 0.0719816, -0.0955347, 0.1860301
1: -0.0760552, 0.2038522, -0.0526925, 0.1818441, -0.2578994, 0.2565446
2: 0.0042095, 0.2710823, 0.0169597, 0.2437519, -0.2395424, 0.2541226
3: -0.0383188, 0.1244072, -0.0247090, 0.1013449, -0.1396637, 0.1491162
4: 0.0045975, 0.2538497, 0.0185986, 0.2299674, -0.2253698, 0.2352512

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0266729, 0.0634494, -0.0870025, 0.1745638
1: -0.0760552, 0.2038522, -0.0436111, 0.1676113, -0.2436665, 0.2474633
2: 0.0042095, 0.2710823, 0.0208580, 0.2285865, -0.2243770, 0.2502243
3: -0.0383188, 0.1244072, -0.0223003, 0.0900128, -0.1283316, 0.1467075
4: 0.0045975, 0.2538497, 0.0212943, 0.2158635, -0.2112660, 0.2325554

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0130985, 0.0806730, -0.1042261, 0.1881382
1: -0.0760552, 0.2038522, -0.0605808, 0.2375910, -0.3136462, 0.2644330
2: 0.0042095, 0.2710823, 0.0148670, 0.3093624, -0.3051528, 0.2562153
3: -0.0383188, 0.1244072, -0.0233690, 0.1373056, -0.1756244, 0.1477762
4: 0.0045975, 0.2538497, 0.0186814, 0.2925131, -0.2879156, 0.2351683

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235531, 0.2012366, 0.0274709, 0.0656589, -0.0892120, 0.1737657
1: -0.0760552, 0.2038522, -0.0460902, 0.1935771, -0.2696323, 0.2499424
2: 0.0042095, 0.2710823, 0.0207678, 0.2596045, -0.2553950, 0.2503145
3: -0.0383188, 0.1244072, -0.0209610, 0.1065546, -0.1448734, 0.1453682
4: 0.0045975, 0.2538497, 0.0222067, 0.2458720, -0.2412745, 0.2316430

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, 0.0152065, 0.0719816, -0.1009345, 0.1880900
1: -0.0891756, 0.2573025, -0.0526925, 0.1818441, -0.2710198, 0.3099950
2: 0.0028422, 0.3348821, 0.0169597, 0.2437519, -0.2409098, 0.3179224
3: -0.0384304, 0.1591823, -0.0247090, 0.1013449, -0.1397753, 0.1838913
4: 0.0068889, 0.3157227, 0.0185986, 0.2299674, -0.2230785, 0.2971241

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0289529, 0.2032966, 0.0266729, 0.0634494, -0.0924022, 0.1766237
1: -0.0891756, 0.2573025, -0.0436111, 0.1676113, -0.2567869, 0.3009136
2: 0.0028422, 0.3348821, 0.0208580, 0.2285865, -0.2257443, 0.3140242
3: -0.0384304, 0.1591823, -0.0223003, 0.0900128, -0.1284432, 0.1814826
4: 0.0068889, 0.3157227, 0.0212943, 0.2158635, -0.2089746, 0.2944283

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, 0.0152065, 0.0719816, -0.0751764, 0.0846196
1: -0.0681015, 0.2029338, -0.0526925, 0.1818441, -0.2499456, 0.2556263
2: 0.0107180, 0.2719691, 0.0169597, 0.2437519, -0.2330339, 0.2550094
3: -0.0302884, 0.1220847, -0.0247090, 0.1013449, -0.1316333, 0.1467937
4: 0.0149963, 0.2578355, 0.0185986, 0.2299674, -0.2149711, 0.2392369

Time for backsubstitution: 2.28 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.20 + 417.70 = 420.90 seconds
