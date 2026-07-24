## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 3600 seconds
Split limit: 100
Threshold: 40.2367993236


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063)
1: (-31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645)
2: (-27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051)
3: (-31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163)
4: (-33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590)
5: (-30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514)
6: (-42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090)
7: (-35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645)
8: (-37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019)
9: (-33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633)
10: (-47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440)
11: (-45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956)
12: (-46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259)
13: (-48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517)
14: (-75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646)
15: (-39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203)
16: (-48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496)
17: (-75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880)
18: (-41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072)
19: (-33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879)
20: (-30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128)
21: (-44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849)
22: (-50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121)
23: (-35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036)
24: (-43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857)
25: (-36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760)
26: (-48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598)
27: (-44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458)
28: (-33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577)
29: (-56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081)
30: (-40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127)
31: (-43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922)
32: (-45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578)
33: (-64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501)
34: (-53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290)
35: (-55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473)
36: (-52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673)
37: (-73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547)
38: (-64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222)
39: (-76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161)
40: (-60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840)
41: (-44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646)
42: (-32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.29 + 122.65 = 124.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -40.2770764, upper bound: 40.2770764

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2345184, upper bound: 40.2679349
time: 91.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2679349, upper bound: 40.2345184
time: 86.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 178.08 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 178.08
Output dim: 4, lower bound: -40.2345184, upper bound: 40.2679349
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 178.08
Output dim: 4, lower bound: -40.2679349, upper bound: 40.2345184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.1356575, upper bound: 40.2659761
time: 89.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2325493, upper bound: 40.1696709
time: 81.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.1696709, upper bound: 40.2325493
time: 101.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2659761, upper bound: 40.1356576
time: 98.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 201.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 201.75
Output dim: 4, lower bound: -40.1356575, upper bound: 40.2659761
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 201.75
Output dim: 4, lower bound: -40.2325493, upper bound: 40.1696709
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 201.75
Output dim: 4, lower bound: -40.1696709, upper bound: 40.2325493
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 201.75
Output dim: 4, lower bound: -40.2659761, upper bound: 40.1356576

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.0832178, upper bound: 40.2634053
time: 81.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.1319966, upper bound: 40.2151402
time: 77.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2151402, upper bound: 40.1319966
time: 81.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2634053, upper bound: 40.0832178
time: 89.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 173.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 173.02
Output dim: 4, lower bound: -40.0832178, upper bound: 40.2634053
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 173.02
Output dim: 4, lower bound: -40.1319966, upper bound: 40.2151402
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 173.02
Output dim: 4, lower bound: -40.2151402, upper bound: 40.1319966
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 173.02
Output dim: 4, lower bound: -40.2634053, upper bound: 40.0832178

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.0403464, upper bound: 40.2633399
time: 94.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.0831507, upper bound: 40.2216222
time: 88.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2216222, upper bound: 40.0831507
time: 75.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2633399, upper bound: 40.0403465
time: 88.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 166.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 166.41
Output dim: 4, lower bound: -40.0403464, upper bound: 40.2633399
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 166.41
Output dim: 4, lower bound: -40.0831507, upper bound: 40.2216222
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 166.41
Output dim: 4, lower bound: -40.2216222, upper bound: 40.0831507
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 166.41
Output dim: 4, lower bound: -40.2633399, upper bound: 40.0403465

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.0392263, upper bound: 40.2252406
time: 99.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.0017797, upper bound: 40.2622328
time: 87.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2622328, upper bound: 40.0017797
time: 234.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2252406, upper bound: 40.0392263
time: 81.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 318.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 318.45
Output dim: 4, lower bound: -40.0392263, upper bound: 40.2252406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 318.45
Output dim: 4, lower bound: -40.0017797, upper bound: 40.2622328
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 318.45
Output dim: 4, lower bound: -40.2622328, upper bound: 40.0017797
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 318.45
Output dim: 4, lower bound: -40.2252406, upper bound: 40.0392263

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -39.9982247, upper bound: 40.2339216
time: 81.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -39.9844294, upper bound: 40.2612519
time: 92.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2612518, upper bound: 39.9844294
time: 83.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2339216, upper bound: 39.9982247
time: 97.18 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 182.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 182.88
Output dim: 4, lower bound: -39.9982247, upper bound: 40.2339216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 182.88
Output dim: 4, lower bound: -39.9844294, upper bound: 40.2612519
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 182.88
Output dim: 4, lower bound: -40.2612518, upper bound: 39.9844294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 182.88
Output dim: 4, lower bound: -40.2339216, upper bound: 39.9982247

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -39.9619093, upper bound: 40.2609883
time: 98.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -39.9839969, upper bound: 40.2368561
time: 113.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2368561, upper bound: 39.9839969
time: 208.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2609883, upper bound: 39.9619093
time: 109.68 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 320.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 320.53
Output dim: 4, lower bound: -39.9619093, upper bound: 40.2609883
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 320.53
Output dim: 4, lower bound: -39.9839969, upper bound: 40.2368561
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 320.53
Output dim: 4, lower bound: -40.2368561, upper bound: 39.9839969
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 320.53
Output dim: 4, lower bound: -40.2609883, upper bound: 39.9619093

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -39.9330676, upper bound: 40.2599880
time: 79.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -39.9608968, upper bound: 40.2322126
time: 151.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063
1: -31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645
2: -27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051
3: -31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163
4: -33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590
5: -30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514
6: -42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090
7: -35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645
8: -37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019
9: -33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633
10: -47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440
11: -45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956
12: -46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259
13: -48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517
14: -75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646
15: -39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203
16: -48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496
17: -75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880
18: -41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072
19: -33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879
20: -30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128
21: -44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849
22: -50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121
23: -35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036
24: -43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857
25: -36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760
26: -48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598
27: -44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458
28: -33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577
29: -56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081
30: -40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127
31: -43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922
32: -45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578
33: -64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501
34: -53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290
35: -55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473
36: -52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673
37: -73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547
38: -64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222
39: -76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161
40: -60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840
41: -44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646
42: -32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -39.9551627, upper bound: 40.2358558
time: 862.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -39.9829835, upper bound: 40.2080321
time: 95.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 960.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 960.01
Output dim: 4, lower bound: -39.9330676, upper bound: 40.2599880
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 960.01
Output dim: 4, lower bound: -39.9608968, upper bound: 40.2322126
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 960.01
Output dim: 4, lower bound: -39.9551627, upper bound: 40.2358558
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 960.01
Output dim: 4, lower bound: -39.9829835, upper bound: 40.2080321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 960.01
Output dim: 4, lower bound: -40.2368561, upper bound: 39.9839969
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 960.01
Output dim: 4, lower bound: -40.2609883, upper bound: 39.9619093

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 124.94 + 3829.47 = 3954.41 seconds
