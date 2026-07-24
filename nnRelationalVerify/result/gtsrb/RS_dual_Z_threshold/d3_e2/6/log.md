## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 44.215418322


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493)
1: (-25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925)
2: (-21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020)
3: (-24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430)
4: (-28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984)
5: (-24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609)
6: (-54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116)
7: (-30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833)
8: (-36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536)
9: (-29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944)
10: (-49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793)
11: (-49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476)
12: (-55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7024841, 79.7024841)
13: (-50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531)
14: (-87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285)
15: (-35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889)
16: (-46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353)
17: (-85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725)
18: (-49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497)
19: (-39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108)
20: (-37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752)
21: (-48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026)
22: (-50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610)
23: (-39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382)
24: (-46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789)
25: (-41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755)
26: (-57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847)
27: (-45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949)
28: (-39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258)
29: (-51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005)
30: (-49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708)
31: (-51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743)
32: (-52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832)
33: (-72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0874100, 106.0874023)
34: (-65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9840164, 81.9840240)
35: (-63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9368591, 85.9368591)
36: (-62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077)
37: (-87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498)
38: (-70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649)
39: (-80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968)
40: (-62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332)
41: (-55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251)
42: (-36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.90 + 95.00 = 97.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -44.2596780, upper bound: 44.2596780

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1658684, upper bound: 44.2554161
time: 79.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2554161, upper bound: 44.1658684
time: 95.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 174.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 174.30
Output dim: 4, lower bound: -44.1658684, upper bound: 44.2554161
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 174.30
Output dim: 4, lower bound: -44.2554161, upper bound: 44.1658684

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6832504, 79.6814880
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0555878, 106.0564194
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9607239, 81.9611969
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9252548, 85.9252167
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0762806, upper bound: 44.2500842
time: 87.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1606171, upper bound: 44.1653447
time: 95.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6814804, 79.6832504
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0564117, 106.0555725
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9611969, 81.9607162
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9252243, 85.9252625
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1653448, upper bound: 44.1606171
time: 83.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2500842, upper bound: 44.0762806
time: 90.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 176.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 176.60
Output dim: 4, lower bound: -44.0762806, upper bound: 44.2500842
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 176.60
Output dim: 4, lower bound: -44.1606171, upper bound: 44.1653447
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 176.60
Output dim: 4, lower bound: -44.1653448, upper bound: 44.1606171
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 176.60
Output dim: 4, lower bound: -44.2500842, upper bound: 44.0762806

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6604004, 79.6527863
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0255890, 106.0276337
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9530029, 81.9548111
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9209671, 85.9207916
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0264437, upper bound: 44.2452469
time: 81.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0711982, upper bound: 44.2014679
time: 88.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6527863, 79.6604004
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0276337, 106.0255890
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9548035, 81.9530106
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9207840, 85.9209595
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2014679, upper bound: 44.0711982
time: 104.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2452469, upper bound: 44.0264437
time: 87.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 194.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 194.96
Output dim: 4, lower bound: -44.0264437, upper bound: 44.2452469
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 194.96
Output dim: 4, lower bound: -44.0711982, upper bound: 44.2014679
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 194.96
Output dim: 4, lower bound: -44.2014679, upper bound: 44.0711982
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 194.96
Output dim: 4, lower bound: -44.2452469, upper bound: 44.0264437

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6359482, 79.6254272
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0248260, 106.0269852
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9584503, 81.9609985
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9321136, 85.9325104
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0219483, upper bound: 44.2255265
time: 77.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0065677, upper bound: 44.2405136
time: 88.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6254196, 79.6359482
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0269928, 106.0248108
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9609985, 81.9584503
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9325104, 85.9321213
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2405137, upper bound: 44.0065677
time: 83.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2255266, upper bound: 44.0219483
time: 90.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 175.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 175.84
Output dim: 4, lower bound: -44.0219483, upper bound: 44.2255265
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 175.84
Output dim: 4, lower bound: -44.0065677, upper bound: 44.2405136
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 175.84
Output dim: 4, lower bound: -44.2405137, upper bound: 44.0065677
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 175.84
Output dim: 4, lower bound: -44.2255266, upper bound: 44.0219483

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6344528, 79.6248474
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0218048, 106.0265427
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9528961, 81.9591293
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9286499, 85.9313965
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9849389, upper bound: 44.2231423
time: 89.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9849389, upper bound: 44.1364961
time: 127.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6359482, 79.6239395
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0248260, 106.0239716
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9584503, 81.9554443
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9321136, 85.9290390
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9695354, upper bound: 44.2381368
time: 92.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9695354, upper bound: 44.1516723
time: 77.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6239548, 79.6353607
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0239716, 106.0243607
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9554443, 81.9565811
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9290466, 85.9310074
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1516724, upper bound: 44.0012532
time: 108.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2381369, upper bound: 43.9695354
time: 79.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6254196, 79.6344604
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0269928, 106.0217972
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9609985, 81.9528961
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9325104, 85.9286575
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1364962, upper bound: 44.0166457
time: 101.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2231424, upper bound: 43.9849388
time: 85.23 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 189.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 189.34
Output dim: 4, lower bound: -43.9849389, upper bound: 44.2231423
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 189.34
Output dim: 4, lower bound: -43.9849389, upper bound: 44.1364961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 189.34
Output dim: 4, lower bound: -43.9695354, upper bound: 44.2381368
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 189.34
Output dim: 4, lower bound: -43.9695354, upper bound: 44.1516723
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 189.34
Output dim: 4, lower bound: -44.1516724, upper bound: 44.0012532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 189.34
Output dim: 4, lower bound: -44.2381369, upper bound: 43.9695354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 189.34
Output dim: 4, lower bound: -44.1364962, upper bound: 44.0166457
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 189.34
Output dim: 4, lower bound: -44.2231424, upper bound: 43.9849388

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6323242, 79.6225204
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0022049, 106.0088654
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9392700, 81.9472733
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9181061, 85.9214172
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9497012, upper bound: 44.2207228
time: 100.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9825908, upper bound: 44.1876789
time: 99.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6338196, 79.6216125
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0052261, 106.0063019
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9448242, 81.9435883
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9215698, 85.9190674
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9342825, upper bound: 44.2357323
time: 72.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9671822, upper bound: 44.2027784
time: 90.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6216125, 79.6332397
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0062943, 106.0047760
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9435883, 81.9429626
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9190674, 85.9204559
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1164713, upper bound: 43.9671822
time: 105.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2357323, upper bound: 43.9342825
time: 86.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6230927, 79.6323318
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0093155, 106.0022125
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9491272, 81.9392700
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9225311, 85.9181061
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1876789, upper bound: 43.9825907
time: 99.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2207228, upper bound: 43.9497012
time: 95.90 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 198.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 198.28
Output dim: 4, lower bound: -43.9497012, upper bound: 44.2207228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 198.28
Output dim: 4, lower bound: -43.9825908, upper bound: 44.1876789
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 198.28
Output dim: 4, lower bound: -43.9342825, upper bound: 44.2357323
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 198.28
Output dim: 4, lower bound: -43.9671822, upper bound: 44.2027784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 198.28
Output dim: 4, lower bound: -44.1164713, upper bound: 43.9671822
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 198.28
Output dim: 4, lower bound: -44.2357323, upper bound: 43.9342825
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 198.28
Output dim: 4, lower bound: -44.1876789, upper bound: 43.9825907
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 198.28
Output dim: 4, lower bound: -44.2207228, upper bound: 43.9497012

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6109161, 79.5954895
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9925385, 106.0007935
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9346619, 81.9431458
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9237289, 85.9264526
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.8945344, upper bound: 44.2193205
time: 92.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8945344, upper bound: 44.1665290
time: 90.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6124115, 79.5945892
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9955444, 105.9982224
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9402161, 81.9394531
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9272079, 85.9240952
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.8790543, upper bound: 44.2343268
time: 90.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9328509, upper bound: 44.1816496
time: 81.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5945892, 79.6118240
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9982300, 105.9951096
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9394531, 81.9383545
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9240952, 85.9260788
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1816497, upper bound: 43.9328509
time: 90.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2343269, upper bound: 43.8790543
time: 93.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5960693, 79.6109161
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0012207, 105.9925461
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9450073, 81.9346619
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9275742, 85.9237213
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1665290, upper bound: 43.9482673
time: 85.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2193205, upper bound: 43.8945344
time: 115.73 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 204.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 204.00
Output dim: 4, lower bound: -43.8945344, upper bound: 44.2193205
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 204.00
Output dim: 4, lower bound: -43.8945344, upper bound: 44.1665290
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 204.00
Output dim: 4, lower bound: -43.8790543, upper bound: 44.2343268
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 204.00
Output dim: 4, lower bound: -43.9328509, upper bound: 44.1816496
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 204.00
Output dim: 4, lower bound: -44.1816497, upper bound: 43.9328509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 204.00
Output dim: 4, lower bound: -44.2343269, upper bound: 43.8790543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 204.00
Output dim: 4, lower bound: -44.1665290, upper bound: 43.9482673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 204.00
Output dim: 4, lower bound: -44.2193205, upper bound: 43.8945344

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6145325, 79.5992126
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9923553, 106.0005417
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9354172, 81.9437408
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9255676, 85.9279327
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8909922, upper bound: 44.1597682
time: 74.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.8418177, upper bound: 44.2166630
time: 91.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6160278, 79.5983124
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9953461, 105.9979706
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9409714, 81.9400482
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9290466, 85.9255829
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8755175, upper bound: 44.1748632
time: 82.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.8263319, upper bound: 44.2316543
time: 80.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5983276, 79.6154480
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9979706, 105.9949188
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9400558, 81.9391098
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9255981, 85.9279175
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2316544, upper bound: 43.8263319
time: 89.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0881378, upper bound: 43.8755175
time: 88.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5998077, 79.6145477
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0009613, 105.9923553
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9455948, 81.9354248
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9290771, 85.9255676
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2166631, upper bound: 43.8418176
time: 111.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1597682, upper bound: 43.8909922
time: 93.86 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 208.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 208.27
Output dim: 4, lower bound: -43.8909922, upper bound: 44.1597682
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 208.27
Output dim: 4, lower bound: -43.8418177, upper bound: 44.2166630
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 208.27
Output dim: 4, lower bound: -43.8755175, upper bound: 44.1748632
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 208.27
Output dim: 4, lower bound: -43.8263319, upper bound: 44.2316543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 208.27
Output dim: 4, lower bound: -44.2316544, upper bound: 43.8263319
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 208.27
Output dim: 4, lower bound: -44.0881378, upper bound: 43.8755175
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 208.27
Output dim: 4, lower bound: -44.2166631, upper bound: 43.8418176
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 208.27
Output dim: 4, lower bound: -44.1597682, upper bound: 43.8909922

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6143341, 79.5987625
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9887543, 105.9932404
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9304276, 81.9336243
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9228668, 85.9224930
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8263903, upper bound: 44.1839370
time: 81.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8078922, upper bound: 44.2003415
time: 77.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6158295, 79.5978546
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9917603, 105.9906769
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9359818, 81.9299316
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9263458, 85.9201431
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.8108750, upper bound: 44.1983009
time: 82.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.7935442, upper bound: 44.2155883
time: 90.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5978546, 79.6152420
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9906769, 105.9913254
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9299393, 81.9341125
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9201355, 85.9252243
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2155884, upper bound: 43.7935442
time: 99.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1983009, upper bound: 43.8108750
time: 67.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5993500, 79.6143341
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9936676, 105.9887543
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9355087, 81.9304199
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9236145, 85.9228668
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2003415, upper bound: 43.8078922
time: 143.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1839371, upper bound: 43.8263902
time: 81.93 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 227.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -43.8263903, upper bound: 44.1839370
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -43.8078922, upper bound: 44.2003415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -43.8108750, upper bound: 44.1983009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 227.40
Output dim: 4, lower bound: -43.7935442, upper bound: 44.2155883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 227.40
Output dim: 4, lower bound: -44.2155884, upper bound: 43.7935442
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -44.1983009, upper bound: 43.8108750
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -44.2003415, upper bound: 43.8078922
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 227.40
Output dim: 4, lower bound: -44.1839371, upper bound: 43.8263902

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6101837, 79.5896301
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9902344, 105.9903793
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9345474, 81.9294281
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9264832, 85.9197006
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 632

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.7769611, upper bound: 44.1077355
time: 154.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.7445315, upper bound: 44.2102562
time: 102.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.5896454, 79.6095886
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -105.9903717, 105.9897842
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9294357, 81.9326782
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9196930, 85.9253540
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 632

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2102563, upper bound: 43.7445315
time: 86.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1077355, upper bound: 43.7769611
time: 74.95 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 163.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 163.73
Output dim: 4, lower bound: -43.7769611, upper bound: 44.1077355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 163.73
Output dim: 4, lower bound: -43.7445315, upper bound: 44.2102562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 163.73
Output dim: 4, lower bound: -44.2102563, upper bound: 43.7445315
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 163.73
Output dim: 4, lower bound: -44.1077355, upper bound: 43.7769611

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 97.90 + 5407.99 = 5505.89 seconds
