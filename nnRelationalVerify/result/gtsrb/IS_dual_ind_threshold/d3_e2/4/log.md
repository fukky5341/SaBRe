## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.181457323


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388)
1: (-26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461)
2: (-24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001)
3: (-28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680)
4: (-31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228)
5: (-28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674)
6: (-55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035)
7: (-32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505)
8: (-37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959)
9: (-30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573)
10: (-49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753)
11: (-48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688)
12: (-59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871)
13: (-51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880)
14: (-79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897)
15: (-38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579)
16: (-48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783)
17: (-79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988)
18: (-48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538)
19: (-38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658)
20: (-34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885)
21: (-46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846)
22: (-49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286)
23: (-37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486)
24: (-45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513)
25: (-39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552)
26: (-56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825)
27: (-46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792)
28: (-37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667)
29: (-51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515)
30: (-46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646)
31: (-49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272)
32: (-55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567993)
33: (-73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0983200)
34: (-63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2917480)
35: (-60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6587524)
36: (-60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986)
37: (-89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701843)
38: (-69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888)
39: (-83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197)
40: (-65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349915)
41: (-58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127)
42: (-40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.88 + 159.06 = 161.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -38.2196770, upper bound: 38.2196770

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2120267, upper bound: 38.1481822
time: 132.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2120267, upper bound: 38.2120264
time: 101.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 233.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 233.84
Output dim: 2, lower bound: -38.2120267, upper bound: 38.1481822
IS_A2, status: Status.UNKNOWN, split count: 1, time: 233.84
Output dim: 2, lower bound: -38.2120267, upper bound: 38.2120264

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -57.1718140, 42.8086052, -57.2110138, 42.9016571, -100.0734711, 100.0196228
1: -26.4862099, 35.1486206, -26.5117359, 35.2362213, -61.7224312, 61.6603546
2: -24.3567314, 36.9197998, -24.3786869, 37.0006561, -61.3573875, 61.2984848
3: -28.6379166, 41.2886658, -28.6565018, 41.3794098, -70.0173264, 69.9451675
4: -31.5895824, 41.1600571, -31.6105957, 41.2442932, -72.8338776, 72.7706528
5: -27.9957561, 42.5802803, -28.0175171, 42.6651649, -70.6609192, 70.5977936
6: -55.2196503, 27.1888199, -55.2513390, 27.2270660, -82.4467163, 82.4401550
7: -32.4038773, 40.3228836, -32.4380341, 40.4708862, -72.8747635, 72.7609177
8: -37.1198349, 49.2701492, -37.1388855, 49.3662872, -86.4861221, 86.4090347
9: -30.2821407, 38.2989197, -30.3014126, 38.3409882, -68.6231308, 68.6003342
10: -49.5296402, 48.1053772, -49.5877533, 48.1481743, -97.6778107, 97.6931305
11: -48.4861679, 29.0227661, -48.5505028, 29.0601959, -77.5463638, 77.5732727
12: -59.5251923, 31.3398705, -59.7574730, 31.3786316, -90.4878845, 90.6814117
13: -51.1980171, 46.9618988, -51.3011284, 47.0028915, -98.2009125, 98.2630310
14: -79.3406982, 42.6545868, -79.5303345, 42.6726646, -122.0133667, 122.1849213
15: -38.0673904, 35.1308746, -38.1230202, 35.1680450, -73.2354355, 73.2538910
16: -48.5900993, 37.0356903, -48.6431274, 37.1760292, -85.7661285, 85.6788177
17: -79.2424316, 34.0989914, -79.4818115, 34.1246109, -113.3670425, 113.5808029
18: -48.1971321, 33.3848991, -48.2697678, 33.4042473, -81.6013794, 81.6546631
19: -38.2614822, 19.2554913, -38.3007889, 19.2651081, -57.5265884, 57.5562820
20: -34.6914558, 24.9678879, -34.7524948, 24.9818840, -59.6733398, 59.7203827
21: -46.1751938, 24.8773918, -46.2265320, 24.8941536, -71.0693512, 71.1039276
22: -48.9542007, 25.1541252, -49.1066704, 25.1801128, -74.1343155, 74.2607956
23: -37.8137245, 26.3360977, -37.8602676, 26.3499203, -64.1636429, 64.1963654
24: -45.4444046, 28.8084984, -45.4848022, 28.8439636, -74.2883682, 74.2933044
25: -39.6264305, 29.4324760, -39.7066193, 29.4537544, -69.0801849, 69.1390991
26: -55.7951202, 38.7271500, -55.9848747, 38.7506447, -94.5457611, 94.7120209
27: -46.0289116, 30.0314026, -46.0712891, 30.0878277, -76.1167374, 76.1026917
28: -36.9681320, 29.8437996, -37.0015907, 29.8558903, -66.8240204, 66.8453903
29: -51.0355377, 24.5578899, -51.1779366, 24.5757484, -75.6112823, 75.7358246
30: -46.3112679, 33.3656082, -46.3442574, 33.4276390, -79.7389069, 79.7098694
31: -49.1127052, 27.7487087, -49.1562195, 27.7695503, -76.8822556, 76.9049301
32: -55.5225868, 24.6066437, -55.5997887, 24.6316910, -80.0333786, 80.0857315
33: -73.7482605, 31.7283802, -73.7873917, 31.8111115, -104.9870377, 104.9126587
34: -63.6997757, 17.8266678, -63.7292709, 17.8759613, -81.1941147, 81.1295929
35: -60.7987137, 24.3201542, -60.8340378, 24.3685417, -84.5626984, 84.5252762
36: -60.7364845, 25.2735825, -60.8392220, 25.2961006, -86.0317688, 86.1119766
37: -89.4178238, 18.5839233, -89.4770889, 18.6112537, -107.8648605, 107.8991852
38: -69.6129761, 29.0387440, -69.7414474, 29.0715942, -98.6845703, 98.7801895
39: -83.3983154, 30.6787186, -83.4362946, 30.7518997, -114.1502151, 114.1150131
40: -65.8093719, 21.3396931, -65.8391724, 21.4514084, -87.2354431, 87.1495895
41: -58.7437286, 28.5929108, -58.7717438, 28.6480904, -87.3918152, 87.3646545
42: -40.1677017, 24.6584053, -40.2236366, 24.6907768, -64.8584747, 64.8820419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087034, upper bound: 38.0807511
time: 111.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2088659, upper bound: 38.1449489
time: 94.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -57.3938637, 42.9760590, -57.2220535, 42.9357834, -100.3296509, 100.1981125
1: -26.6577587, 35.3051567, -26.5186520, 35.2700577, -61.9278183, 61.8238068
2: -24.5485287, 37.0604858, -24.3840237, 37.0314827, -61.5800095, 61.4445114
3: -28.8327713, 41.4618073, -28.6602898, 41.4137306, -70.2465057, 70.1221008
4: -31.7659378, 41.3111191, -31.6133652, 41.2756577, -73.0415955, 72.9244843
5: -28.1658974, 42.7399559, -28.0229855, 42.6964455, -70.8623428, 70.7629395
6: -55.2980270, 27.2909222, -55.2589493, 27.2323608, -82.5303879, 82.5498734
7: -32.6945114, 40.5707054, -32.4473724, 40.5277786, -73.2222900, 73.0180817
8: -37.2880859, 49.4514732, -37.1411667, 49.4029236, -86.6910095, 86.5926361
9: -30.3925438, 38.4150696, -30.3052483, 38.3503647, -68.7429047, 68.7203217
10: -49.6660767, 48.3442917, -49.6087265, 48.1602554, -97.8263321, 97.9530182
11: -48.5893707, 29.0915909, -48.5699158, 29.0508747, -77.6402435, 77.6615067
12: -59.9115448, 31.8487186, -59.8476753, 31.3893661, -90.8765564, 91.2928772
13: -51.3968315, 47.2209282, -51.3361893, 47.0148773, -98.4117126, 98.5571136
14: -79.7112427, 43.0292053, -79.6017532, 42.6776581, -122.3889008, 122.6309586
15: -38.1935196, 35.2568512, -38.1195984, 35.1787262, -73.3722458, 73.3764496
16: -48.8921242, 37.2880096, -48.6582832, 37.2182388, -86.1103668, 85.9462891
17: -79.6671448, 34.4916916, -79.5749359, 34.1299973, -113.7971420, 114.0666275
18: -48.3900871, 33.4641800, -48.2944221, 33.4077454, -81.7978363, 81.7586060
19: -38.3456268, 19.2588215, -38.3141937, 19.2510414, -57.5966682, 57.5730133
20: -34.8364792, 25.0689201, -34.7739487, 24.9848232, -59.8213043, 59.8428688
21: -46.2878418, 24.9164658, -46.2437630, 24.8808708, -71.1687164, 71.1602325
22: -49.2303391, 25.3956738, -49.1588707, 25.1862068, -74.4165497, 74.5545425
23: -37.9262695, 26.3458176, -37.8768425, 26.3411751, -64.2674408, 64.2226562
24: -45.6047630, 28.8789978, -45.4969978, 28.8533096, -74.4580688, 74.3759918
25: -39.7902374, 29.5718384, -39.7349091, 29.4594612, -69.2496948, 69.3067474
26: -56.1602135, 39.0910530, -56.0579758, 38.7561035, -94.9163208, 95.1490326
27: -46.2083015, 30.1258736, -46.0834579, 30.1073513, -76.3156509, 76.2093353
28: -37.0704193, 29.8732414, -37.0131302, 29.8509254, -66.9213409, 66.8863678
29: -51.2947197, 24.8092766, -51.2307739, 24.5798645, -75.8745880, 76.0400543
30: -46.4290581, 33.4869537, -46.3529549, 33.4370270, -79.8660889, 79.8399048
31: -49.2462082, 27.7932510, -49.1703110, 27.7637291, -77.0099335, 76.9635620
32: -55.6707840, 24.7737675, -55.6267281, 24.6367588, -80.1846466, 80.2845535
33: -73.9594574, 31.8953819, -73.7979889, 31.8397961, -105.3083344, 105.0866318
34: -63.8072357, 17.9348869, -63.7371330, 17.8925285, -81.4538116, 81.2488022
35: -60.9144859, 24.4059772, -60.8447342, 24.3835926, -84.7696533, 84.6212311
36: -60.9387283, 25.4797440, -60.8778343, 25.3027000, -86.2404175, 86.3572693
37: -89.5943146, 18.6509113, -89.4959259, 18.6134472, -108.0445251, 107.9896011
38: -69.8765945, 29.2488174, -69.7895813, 29.0812969, -98.9578934, 99.0383987
39: -83.5216675, 30.8039474, -83.4464111, 30.7747612, -114.2964325, 114.2503586
40: -65.9757538, 21.5110435, -65.8452606, 21.4889297, -87.4604950, 87.3241425
41: -58.8555412, 28.6963654, -58.7793846, 28.6618156, -87.5173569, 87.4757538
42: -40.2780838, 24.8038406, -40.2422981, 24.6983204, -64.9764023, 65.0461426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087034, upper bound: 38.1446980
time: 141.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2088659, upper bound: 38.2088653
time: 70.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 214.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 214.28
Output dim: 2, lower bound: -38.2087034, upper bound: 38.0807511
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 214.28
Output dim: 2, lower bound: -38.2088659, upper bound: 38.1449489
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 214.28
Output dim: 2, lower bound: -38.2087034, upper bound: 38.1446980
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 214.28
Output dim: 2, lower bound: -38.2088659, upper bound: 38.2088653

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -57.1333504, 42.7905502, -57.0915031, 42.8448334, -99.9781799, 99.8820496
1: -26.4617672, 35.1380272, -26.4346485, 35.2028618, -61.6646271, 61.5726776
2: -24.3054276, 36.9096680, -24.2161999, 36.9686623, -61.2740898, 61.1258698
3: -28.5925007, 41.2682114, -28.5135574, 41.3151627, -69.9076614, 69.7817688
4: -31.5290527, 41.1443253, -31.4201431, 41.1948166, -72.7238693, 72.5644684
5: -27.9525604, 42.5602226, -27.8813553, 42.6018410, -70.5543976, 70.4415741
6: -55.1941910, 27.1464405, -55.1708069, 27.1007767, -82.2949677, 82.3172455
7: -32.3654633, 40.3082848, -32.3199959, 40.4249115, -72.7903748, 72.6282806
8: -37.0634804, 49.2525177, -36.9602737, 49.3106842, -86.3741608, 86.2127914
9: -30.2630463, 38.2377739, -30.2410603, 38.1477661, -68.4108124, 68.4788361
10: -49.5017853, 47.9828262, -49.5000153, 47.7594070, -97.2611923, 97.4828415
11: -48.4606247, 28.9428730, -48.4693871, 28.8076134, -77.2682343, 77.4122620
12: -59.5044708, 31.2189465, -59.6927643, 30.9949341, -90.0811157, 90.4940948
13: -51.1794510, 46.9035530, -51.2426147, 46.8190002, -97.9984512, 98.1461639
14: -79.3101578, 42.5462189, -79.4351273, 42.3293610, -121.6395187, 121.9813461
15: -37.9992867, 35.1086349, -37.9074707, 35.0973511, -73.0966339, 73.0161057
16: -48.5563431, 36.9632874, -48.5360985, 36.9471855, -85.5035248, 85.4993896
17: -79.2233734, 34.0140457, -79.4224091, 33.8577271, -113.0811005, 113.4364548
18: -48.1613007, 33.3601036, -48.1580658, 33.3259277, -81.4872284, 81.5181732
19: -38.2377472, 19.2461205, -38.2258759, 19.2356644, -57.4734116, 57.4719963
20: -34.6713028, 24.9422493, -34.6889420, 24.9008904, -59.5721931, 59.6311913
21: -46.1512184, 24.8502693, -46.1505051, 24.8086109, -70.9598312, 71.0007782
22: -48.9064713, 25.1296501, -48.9560127, 25.1018620, -74.0083313, 74.0856628
23: -37.7952576, 26.3228989, -37.8020172, 26.3082161, -64.1034698, 64.1249161
24: -45.3881378, 28.7984524, -45.3096275, 28.8125191, -74.2006531, 74.1080780
25: -39.5991516, 29.4150505, -39.6216660, 29.3985138, -68.9976654, 69.0367126
26: -55.7677345, 38.6762161, -55.8989029, 38.5908432, -94.3585815, 94.5751190
27: -45.9577408, 30.0212288, -45.8462753, 30.0558071, -76.0135498, 75.8675079
28: -36.9393005, 29.8298721, -36.9123306, 29.8118992, -66.7511978, 66.7422028
29: -51.0045395, 24.5290546, -51.0800476, 24.4857197, -75.4902573, 75.6091003
30: -46.2886086, 33.3267822, -46.2727318, 33.3113480, -79.5999603, 79.5995178
31: -49.0774956, 27.7343292, -49.0468597, 27.7241936, -76.8016891, 76.7811890
32: -55.4994469, 24.5567780, -55.5267830, 24.4772453, -79.8542175, 79.9613800
33: -73.6780548, 31.7092152, -73.5648117, 31.7504864, -104.8481598, 104.6452332
34: -63.6635704, 17.8095264, -63.6148758, 17.8217545, -81.0906754, 80.9610596
35: -60.7453232, 24.3054352, -60.6646957, 24.3221893, -84.4546967, 84.3187256
36: -60.7045898, 25.2606277, -60.7411346, 25.2547531, -85.9582291, 86.0005798
37: -89.3691254, 18.5662651, -89.3250885, 18.5554466, -107.7537003, 107.7193222
38: -69.5800781, 29.0204430, -69.6388474, 29.0137253, -98.5938034, 98.6592865
39: -83.3529205, 30.6671734, -83.2944260, 30.7153187, -114.0682373, 113.9616013
40: -65.7660065, 21.3285885, -65.7027359, 21.4176788, -87.1539841, 86.9934387
41: -58.7178879, 28.5771065, -58.6896248, 28.5992088, -87.3170929, 87.2667313
42: -40.1468239, 24.6015167, -40.1574707, 24.5200386, -64.6668625, 64.7589874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
time: 109.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
time: 88.21 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -57.1627274, 42.8034058, -57.2622986, 42.9400482, -100.1027756, 100.0657043
1: -26.4800644, 35.1469650, -26.5423336, 35.2923050, -61.7723694, 61.6893005
2: -24.3488140, 36.9174957, -24.3991909, 37.1152229, -61.4640350, 61.3166885
3: -28.6258869, 41.2840347, -28.6631069, 41.4740601, -70.0999451, 69.9471436
4: -31.5717163, 41.1564941, -31.6159630, 41.3959198, -72.9676361, 72.7724609
5: -27.9842701, 42.5761604, -28.0322762, 42.7589493, -70.7432175, 70.6084366
6: -55.2142830, 27.1711807, -55.3510818, 27.2255077, -82.4397888, 82.5222626
7: -32.3956718, 40.3199310, -32.4947510, 40.5173721, -72.9130402, 72.8146820
8: -37.1113739, 49.2667999, -37.1646957, 49.5075035, -86.6188812, 86.4314957
9: -30.2778130, 38.2903748, -30.4358501, 38.3610764, -68.6388855, 68.7262268
10: -49.5248032, 48.0896873, -49.9565849, 48.1532860, -97.6780853, 98.0462723
11: -48.4814224, 29.0122147, -48.8621407, 29.0599232, -77.5413437, 77.8743591
12: -59.5209770, 31.3246956, -60.0956192, 31.3956108, -90.4970703, 91.0071259
13: -51.1920433, 46.9524612, -51.3598976, 47.0555420, -98.2475891, 98.3123627
14: -79.3333282, 42.6418495, -79.8024673, 42.6745148, -122.0078430, 122.4443207
15: -38.0583572, 35.1270370, -38.1557999, 35.3372154, -73.3955688, 73.2828369
16: -48.5828590, 37.0255318, -48.8624611, 37.1840401, -85.7668991, 85.8879929
17: -79.2378845, 34.0894699, -79.7396545, 34.1657372, -113.4036255, 113.8291245
18: -48.1849823, 33.3770981, -48.3487854, 33.4282150, -81.6131973, 81.7258835
19: -38.2550583, 19.2528858, -38.3660431, 19.2840557, -57.5391159, 57.6189270
20: -34.6881943, 24.9633675, -34.8398285, 24.9867096, -59.6749039, 59.8031960
21: -46.1704369, 24.8713150, -46.3757553, 24.9054718, -71.0759125, 71.2470703
22: -48.9345284, 25.1490860, -49.1153488, 25.2588081, -74.1933365, 74.2644348
23: -37.8102264, 26.3285313, -37.9260178, 26.3560867, -64.1663132, 64.2545471
24: -45.4351845, 28.8054314, -45.5110741, 28.9094486, -74.3446350, 74.3165054
25: -39.6188126, 29.4283733, -39.7335587, 29.4951706, -69.1139832, 69.1619339
26: -55.7896652, 38.7147636, -56.0758438, 38.7689056, -94.5585709, 94.7906036
27: -46.0197906, 30.0278511, -46.1088448, 30.1971016, -76.2168884, 76.1366959
28: -36.9619293, 29.8388672, -37.0277939, 29.9162712, -66.8782043, 66.8666611
29: -51.0261002, 24.5517426, -51.2213097, 24.6007729, -75.6268768, 75.7730560
30: -46.3067589, 33.3549194, -46.4021912, 33.4426117, -79.7493744, 79.7571106
31: -49.1031876, 27.7440567, -49.2353439, 27.7826614, -76.8858490, 76.9794006
32: -55.5182915, 24.5988216, -55.7183418, 24.6414604, -80.0365295, 80.1988068
33: -73.7381058, 31.7250824, -73.8008118, 32.0124741, -105.2051926, 104.9062653
34: -63.6936302, 17.8221703, -63.7489624, 18.0114098, -81.3621368, 81.1224060
35: -60.7910347, 24.3177433, -60.8461838, 24.5647697, -84.7789536, 84.5195847
36: -60.7292557, 25.2708321, -60.8649597, 25.3776379, -86.1062851, 86.1348038
37: -89.4054337, 18.5805855, -89.5220566, 18.6740971, -107.9149475, 107.9341812
38: -69.6048355, 29.0342941, -69.7985535, 29.1395588, -98.7443924, 98.8328476
39: -83.3837662, 30.6762314, -83.4741669, 30.8681850, -114.2519531, 114.1503983
40: -65.8024826, 21.3332291, -65.9057159, 21.4859161, -87.2657623, 87.2062759
41: -58.7387505, 28.5859566, -58.8247299, 28.6621914, -87.4009399, 87.4106903
42: -40.1634903, 24.6381855, -40.3976746, 24.6838169, -64.8473053, 65.0358582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1253291
time: 93.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1424323
time: 120.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -57.3553772, 42.9580994, -57.1025887, 42.8789825, -100.2343597, 100.0606842
1: -26.6331711, 35.2946625, -26.4415188, 35.2367096, -61.8698807, 61.7361832
2: -24.4971676, 37.0504036, -24.2215385, 36.9994926, -61.4966583, 61.2719421
3: -28.7873287, 41.4415169, -28.5173149, 41.3494949, -70.1368256, 69.9588318
4: -31.7053223, 41.2954330, -31.4229279, 41.2261429, -72.9314651, 72.7183609
5: -28.1226120, 42.7200012, -27.8868256, 42.6331177, -70.7557297, 70.6068268
6: -55.2725601, 27.2483406, -55.1782913, 27.1059685, -82.3785248, 82.4266357
7: -32.6560287, 40.5562210, -32.3293495, 40.4818573, -73.1378860, 72.8855743
8: -37.2316895, 49.4339752, -36.9625511, 49.3473663, -86.5790558, 86.3965302
9: -30.3732128, 38.3540077, -30.2448521, 38.1573563, -68.5305710, 68.5988617
10: -49.6386032, 48.2215347, -49.5215263, 47.7716370, -97.4102402, 97.7430573
11: -48.5636635, 29.0119820, -48.4885635, 28.7983322, -77.3619995, 77.5005493
12: -59.8910522, 31.7275772, -59.7829590, 31.0056839, -90.4699707, 91.1053848
13: -51.3783531, 47.1623116, -51.2776642, 46.8309517, -98.2093048, 98.4399719
14: -79.6807404, 42.9207611, -79.5065994, 42.3343277, -122.0150681, 122.4273605
15: -38.1257935, 35.2341232, -37.9039688, 35.1078720, -73.2336655, 73.1380920
16: -48.8579369, 37.2157822, -48.5511398, 36.9896507, -85.8475876, 85.7669220
17: -79.6481171, 34.4065170, -79.5155716, 33.8630791, -113.5112000, 113.9220886
18: -48.3541412, 33.4393272, -48.1826897, 33.3293686, -81.6835098, 81.6220169
19: -38.3218765, 19.2494774, -38.2391968, 19.2216148, -57.5434914, 57.4886742
20: -34.8164062, 25.0432549, -34.7103806, 24.9038258, -59.7202301, 59.7536354
21: -46.2638206, 24.8893318, -46.1675911, 24.7953377, -71.0591583, 71.0569229
22: -49.1827698, 25.3708878, -49.0084267, 25.1078434, -74.2906113, 74.3793182
23: -37.9078217, 26.3326359, -37.8185463, 26.2994480, -64.2072678, 64.1511841
24: -45.5482674, 28.8690147, -45.3219299, 28.8218479, -74.3701172, 74.1909485
25: -39.7631187, 29.5543404, -39.6505089, 29.4041634, -69.1672821, 69.2048492
26: -56.1330643, 39.0400162, -55.9720459, 38.5962372, -94.7293015, 95.0120621
27: -46.1368942, 30.1157112, -45.8584061, 30.0753117, -76.2122040, 75.9741211
28: -37.0414696, 29.8592968, -36.9238892, 29.8068542, -66.8483276, 66.7831879
29: -51.2637634, 24.7803154, -51.1329193, 24.4897938, -75.7535553, 75.9132385
30: -46.4063263, 33.4481735, -46.2813339, 33.3208084, -79.7271347, 79.7295074
31: -49.2107124, 27.7789230, -49.0609932, 27.7183762, -76.9290924, 76.8399200
32: -55.6476746, 24.7237968, -55.5536652, 24.4823532, -80.0055695, 80.1600800
33: -73.8891602, 31.8762493, -73.5753937, 31.7792263, -105.1692429, 104.8193207
34: -63.7709236, 17.9177818, -63.6226234, 17.8383179, -81.3501892, 81.0804367
35: -60.8610039, 24.3913479, -60.6753922, 24.3371830, -84.6615067, 84.4147034
36: -60.9069023, 25.4666290, -60.7798920, 25.2613220, -86.1668549, 86.2458191
37: -89.5456543, 18.6332550, -89.3439484, 18.5574684, -107.9331055, 107.8097763
38: -69.8438339, 29.2304649, -69.6871338, 29.0233288, -98.8671646, 98.9176025
39: -83.4762039, 30.7924290, -83.3045273, 30.7382908, -114.2144928, 114.0969543
40: -65.9321671, 21.5000839, -65.7087708, 21.4554806, -87.3790436, 87.1681137
41: -58.8295937, 28.6805897, -58.6971245, 28.6129837, -87.4425812, 87.3777161
42: -40.2571869, 24.7468300, -40.1760674, 24.5276985, -64.7848816, 64.9228973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1210255
time: 74.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1394054
time: 85.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -57.3848038, 42.9707489, -57.2732620, 42.9742050, -100.3590088, 100.2440109
1: -26.6515617, 35.3035126, -26.5491314, 35.3261986, -61.9777603, 61.8526459
2: -24.5406017, 37.0582085, -24.4044800, 37.1460724, -61.6866760, 61.4626884
3: -28.8207741, 41.4572182, -28.6668320, 41.5083847, -70.3291626, 70.1240540
4: -31.7480755, 41.3075790, -31.6186333, 41.4273338, -73.1754074, 72.9262085
5: -28.1544018, 42.7358246, -28.0377331, 42.7902756, -70.9446793, 70.7735596
6: -55.2926369, 27.2733555, -55.3587303, 27.2304688, -82.5231018, 82.6320877
7: -32.6863213, 40.5677643, -32.5039902, 40.5743103, -73.2606354, 73.0717545
8: -37.2796555, 49.4481583, -37.1669006, 49.5442543, -86.8239136, 86.6150589
9: -30.3882103, 38.4065552, -30.4396973, 38.3706894, -68.7588959, 68.8462524
10: -49.6612549, 48.3285179, -49.9778862, 48.1654015, -97.8266602, 98.3064041
11: -48.5846176, 29.0810852, -48.8814049, 29.0506172, -77.6352386, 77.9624939
12: -59.9073944, 31.8335571, -60.1858521, 31.4062462, -90.8856888, 91.6186371
13: -51.3908806, 47.2114410, -51.3949203, 47.0673866, -98.4582672, 98.6063614
14: -79.7039032, 43.0165482, -79.8738937, 42.6794167, -122.3833160, 122.8904419
15: -38.1845245, 35.2529907, -38.1522484, 35.3478546, -73.5323792, 73.4052429
16: -48.8848534, 37.2778816, -48.8775711, 37.2264023, -86.1112518, 86.1554565
17: -79.6625824, 34.4820518, -79.8327789, 34.1710587, -113.8336411, 114.3148346
18: -48.3778458, 33.4564095, -48.3732834, 33.4316177, -81.8094635, 81.8296967
19: -38.3391151, 19.2562408, -38.3793755, 19.2700367, -57.6091537, 57.6356163
20: -34.8332405, 25.0644226, -34.8613205, 24.9896393, -59.8228798, 59.9257431
21: -46.2830582, 24.9103699, -46.3929634, 24.8921890, -71.1752472, 71.3033295
22: -49.2107925, 25.3905811, -49.1676559, 25.2648830, -74.4756775, 74.5582352
23: -37.9227524, 26.3382721, -37.9425888, 26.3472557, -64.2700043, 64.2808609
24: -45.5954933, 28.8759766, -45.5232887, 28.9188309, -74.5143280, 74.3992615
25: -39.7826042, 29.5677509, -39.7622604, 29.5008469, -69.2834473, 69.3300095
26: -56.1548119, 39.0787125, -56.1489601, 38.7742958, -94.9291077, 95.2276764
27: -46.1991615, 30.1223011, -46.1208763, 30.2166195, -76.4157791, 76.2431793
28: -37.0641899, 29.8683052, -37.0392838, 29.9113140, -66.9755020, 66.9075928
29: -51.2852249, 24.8031387, -51.2741051, 24.6047974, -75.8900223, 76.0772400
30: -46.4245682, 33.4763336, -46.4108734, 33.4519272, -79.8764954, 79.8872070
31: -49.2365646, 27.7885418, -49.2495079, 27.7767887, -77.0133514, 77.0380478
32: -55.6664658, 24.7659302, -55.7452850, 24.6464806, -80.1877136, 80.3976440
33: -73.9493332, 31.8920727, -73.8113861, 32.0412407, -105.5264740, 105.0802460
34: -63.8010864, 17.9304066, -63.7567062, 18.0280018, -81.6217804, 81.2417908
35: -60.9067993, 24.4035721, -60.8568573, 24.5797729, -84.9859161, 84.6155167
36: -60.9315186, 25.4770088, -60.9035988, 25.3843880, -86.3151093, 86.3801041
37: -89.5819244, 18.6475334, -89.5408020, 18.6762085, -108.0949020, 108.0244598
38: -69.8684464, 29.2443752, -69.8467026, 29.1491642, -99.0176086, 99.0910797
39: -83.5070343, 30.8014374, -83.4842300, 30.8913498, -114.3983841, 114.2856674
40: -65.9688721, 21.5045891, -65.9116745, 21.5237751, -87.4909973, 87.3806992
41: -58.8505249, 28.6895218, -58.8323174, 28.6759186, -87.5264435, 87.5218353
42: -40.2738647, 24.7836189, -40.4164276, 24.6912632, -64.9651260, 65.2000427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1882480
time: 85.18 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.2063348
time: 98.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 185.65 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1253291
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1424323
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1210255
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1394054
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1882480
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 185.65
Output dim: 2, lower bound: -38.0770229, upper bound: 38.2063348

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -57.1059990, 42.8884697, -57.1836281, 42.9537086, -100.0597076, 100.0720978
1: -26.4432182, 35.2417831, -26.4837456, 35.3120956, -61.7553139, 61.7255287
2: -24.2057724, 36.9819069, -24.2981873, 37.1323433, -61.3381157, 61.2800941
3: -28.4597034, 41.3324966, -28.5501957, 41.4842644, -69.9439697, 69.8826904
4: -31.3821983, 41.2038727, -31.5033817, 41.4079819, -72.7901764, 72.7072525
5: -27.7985859, 42.6186905, -27.9245224, 42.7688942, -70.5674820, 70.5432129
6: -55.1866837, 27.1133766, -55.3242073, 27.1814041, -82.3680878, 82.4375839
7: -32.3610420, 40.4894257, -32.4030533, 40.5576553, -72.9187012, 72.8924789
8: -36.9481926, 49.3445129, -37.0629921, 49.5233421, -86.4715347, 86.4075012
9: -30.2740803, 38.1548080, -30.4080296, 38.2919617, -68.5660400, 68.5628357
10: -49.4609756, 47.7739182, -49.9380722, 47.9897957, -97.4507751, 97.7119904
11: -48.4134636, 28.7050972, -48.8482361, 28.9306622, -77.3441238, 77.5533295
12: -59.7273483, 31.2143860, -60.1595001, 31.2104015, -90.5066833, 90.9693756
13: -51.2653008, 47.0509987, -51.3560410, 47.0188293, -98.2841339, 98.4070435
14: -79.4731445, 42.5731621, -79.8206940, 42.5385132, -122.0116577, 122.3938599
15: -37.8922348, 35.1547165, -38.0602722, 35.3155975, -73.2078323, 73.2149887
16: -48.7096443, 36.9975471, -48.8290939, 37.1373405, -85.8469849, 85.8266449
17: -79.5051117, 34.1225662, -79.8005371, 34.0571938, -113.5623016, 113.9231033
18: -48.2384491, 33.2279129, -48.3362999, 33.3575516, -81.5960007, 81.5642090
19: -38.2265053, 19.1292477, -38.3501663, 19.2299042, -57.4564095, 57.4794159
20: -34.7098236, 24.9226608, -34.8300095, 24.9445229, -59.6543465, 59.7526703
21: -46.1399307, 24.6780815, -46.3614197, 24.8189678, -70.9589005, 71.0395050
22: -49.0839424, 25.2142048, -49.1301804, 25.2117691, -74.2957153, 74.3443832
23: -37.8211098, 26.2161465, -37.9181976, 26.3083725, -64.1294861, 64.1343460
24: -45.4660835, 28.8166275, -45.4840088, 28.9006386, -74.3667221, 74.3006363
25: -39.6957550, 29.4058475, -39.7365303, 29.4510880, -69.1468430, 69.1423798
26: -55.9896469, 38.6339417, -56.1124039, 38.6311760, -94.6208191, 94.7463455
27: -46.0081177, 30.0789986, -46.0620995, 30.2037315, -76.2118530, 76.1410980
28: -36.9703598, 29.7935715, -37.0109863, 29.8896618, -66.8600235, 66.8045578
29: -51.1884651, 24.5691032, -51.2451706, 24.5295067, -75.7179718, 75.8142700
30: -46.3203125, 33.2412071, -46.3835564, 33.3756104, -79.6959229, 79.6247635
31: -49.0817795, 27.6419411, -49.2078896, 27.7301216, -76.8119049, 76.8498306
32: -55.5632782, 24.5648441, -55.7165375, 24.5808716, -80.0168228, 80.1683502
33: -73.6895752, 31.7488785, -73.7295685, 32.0084076, -105.2120819, 104.8495636
34: -63.6424789, 17.8164043, -63.7069626, 17.9994926, -81.3975983, 81.0611801
35: -60.7084999, 24.2964077, -60.7938309, 24.5576992, -84.7338409, 84.4310303
36: -60.8150444, 25.3982944, -60.8673134, 25.3627186, -86.1766205, 86.2647934
37: -89.4445648, 18.4878483, -89.4984894, 18.6293392, -107.9033051, 107.8136444
38: -69.6940765, 29.1538658, -69.7937164, 29.1246071, -98.8186798, 98.9475861
39: -83.3548889, 30.7132034, -83.4389191, 30.8703003, -114.2251892, 114.1521225
40: -65.8102798, 21.4301014, -65.8648758, 21.5024986, -87.3052826, 87.2546997
41: -58.7452164, 28.5774593, -58.7983894, 28.6424446, -87.3876648, 87.3758469
42: -40.1610870, 24.5093803, -40.3906174, 24.6027222, -64.7638092, 64.8999939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0723830
time: 81.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1851301
time: 77.33 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -57.3780136, 42.9693108, -57.2732620, 42.9742050, -100.3522186, 100.2425690
1: -26.6460037, 35.3023720, -26.5491314, 35.3261986, -61.9722023, 61.8515015
2: -24.5326538, 37.0565643, -24.4044800, 37.1460724, -61.6787262, 61.4610443
3: -28.8138313, 41.4545822, -28.6668320, 41.5083847, -70.3222198, 70.1214142
4: -31.7397614, 41.3051224, -31.6186333, 41.4273338, -73.1670990, 72.9237518
5: -28.1476936, 42.7334175, -28.0377331, 42.7902756, -70.9379730, 70.7711487
6: -55.2889824, 27.2576046, -55.3587303, 27.2304688, -82.5194550, 82.6163330
7: -32.6788292, 40.5661087, -32.5039902, 40.5743103, -73.2531433, 73.0700989
8: -37.2721672, 49.4459534, -37.1669006, 49.5442543, -86.8164215, 86.6128540
9: -30.3847580, 38.4018707, -30.4396973, 38.3706894, -68.7554474, 68.8415680
10: -49.6579819, 48.3149719, -49.9778862, 48.1654015, -97.8233795, 98.2928619
11: -48.5813370, 29.0708046, -48.8814049, 29.0506172, -77.6319580, 77.9522095
12: -59.9048233, 31.8194466, -60.1858521, 31.4062462, -90.8853149, 91.6043472
13: -51.3782959, 47.2074966, -51.3949203, 47.0673866, -98.4456787, 98.6024170
14: -79.6990051, 43.0064087, -79.8738937, 42.6794167, -122.3784180, 122.8803024
15: -38.1694374, 35.2501831, -38.1522484, 35.3478546, -73.5172882, 73.4024353
16: -48.8793602, 37.2686119, -48.8775711, 37.2264023, -86.1057587, 86.1461792
17: -79.6598511, 34.4753799, -79.8327789, 34.1710587, -113.8309097, 114.3081589
18: -48.3746796, 33.4516983, -48.3732834, 33.4316177, -81.8062973, 81.8249817
19: -38.3363342, 19.2536545, -38.3793755, 19.2700367, -57.6063690, 57.6330299
20: -34.8308487, 25.0604877, -34.8613205, 24.9896393, -59.8204880, 59.9218063
21: -46.2798462, 24.9052067, -46.3929634, 24.8921890, -71.1720352, 71.2981720
22: -49.2038231, 25.3829727, -49.1676559, 25.2648830, -74.4687042, 74.5506287
23: -37.9201965, 26.3319588, -37.9425888, 26.3472557, -64.2674561, 64.2745514
24: -45.5884590, 28.8704815, -45.5232887, 28.9188309, -74.5072937, 74.3937683
25: -39.7794113, 29.5638809, -39.7622604, 29.5008469, -69.2802582, 69.3261414
26: -56.1514702, 39.0705795, -56.1489601, 38.7742958, -94.9257660, 95.2195435
27: -46.1937866, 30.1207924, -46.1208763, 30.2166195, -76.4104080, 76.2416687
28: -37.0615540, 29.8638687, -37.0392838, 29.9113140, -66.9728699, 66.9031525
29: -51.2817230, 24.7983589, -51.2741051, 24.6047974, -75.8865204, 76.0724640
30: -46.4214897, 33.4705772, -46.4108734, 33.4519272, -79.8734131, 79.8814545
31: -49.2330246, 27.7836189, -49.2495079, 27.7767887, -77.0098114, 77.0331268
32: -55.6633530, 24.7606144, -55.7452850, 24.6464806, -80.1856308, 80.3922577
33: -73.9423828, 31.8893757, -73.8113861, 32.0412407, -105.5132446, 105.0775681
34: -63.7966080, 17.9278870, -63.7567062, 18.0280018, -81.6081543, 81.2391663
35: -60.9011383, 24.4019680, -60.8568573, 24.5797729, -84.9734344, 84.6137543
36: -60.9256935, 25.4752502, -60.9035988, 25.3843880, -86.3092346, 86.3783340
37: -89.5768890, 18.6398735, -89.5408020, 18.6762085, -108.0902557, 108.0153275
38: -69.8615341, 29.2404003, -69.8467026, 29.1491642, -99.0106964, 99.0871048
39: -83.5015640, 30.7995338, -83.4842300, 30.8913498, -114.3929138, 114.2837677
40: -65.9643097, 21.4977913, -65.9116745, 21.5237751, -87.4856567, 87.3734894
41: -58.8472366, 28.6812801, -58.8323174, 28.6759186, -87.5231552, 87.5135956
42: -40.2709961, 24.7745323, -40.4164276, 24.6912632, -64.9622574, 65.1909637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0770224
time: 190.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.2063348
time: 145.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 338.06 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 338.06
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0723830
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 338.06
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1851301
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 338.06
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0770224
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 338.06
Output dim: 2, lower bound: -38.1882486, upper bound: 38.2063348

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -57.1056175, 42.8883629, -57.1766891, 42.9518127, -100.0574341, 100.0650482
1: -26.4429016, 35.2417068, -26.4779892, 35.3108597, -61.7537613, 61.7196960
2: -24.2052937, 36.9818230, -24.2897701, 37.1309586, -61.3362503, 61.2715912
3: -28.4591599, 41.3323593, -28.5407257, 41.4817734, -69.9409332, 69.8730850
4: -31.3817272, 41.2037582, -31.4950027, 41.4058647, -72.7875900, 72.6987610
5: -27.7980652, 42.6185684, -27.9153728, 42.7667122, -70.5647736, 70.5339432
6: -55.1865082, 27.1125832, -55.3211136, 27.1678619, -82.3543701, 82.4337006
7: -32.3605614, 40.4893341, -32.3939095, 40.5559464, -72.9165039, 72.8832397
8: -36.9477310, 49.3443832, -37.0550156, 49.5211754, -86.4689026, 86.3993988
9: -30.2739029, 38.1544113, -30.4050102, 38.2849274, -68.5588303, 68.5594177
10: -49.4607887, 47.7733154, -49.9349251, 47.9786873, -97.4394760, 97.7082367
11: -48.4132462, 28.7042274, -48.8446732, 28.9157314, -77.3289795, 77.5489044
12: -59.7272034, 31.2135220, -60.1569595, 31.1955490, -90.4913025, 90.9721680
13: -51.2645378, 47.0507622, -51.3425522, 47.0147209, -98.2792587, 98.3933105
14: -79.4728546, 42.5726624, -79.8154602, 42.5291023, -122.0019531, 122.3881226
15: -37.8907776, 35.1545410, -38.0358810, 35.3125534, -73.2033310, 73.1904221
16: -48.7093735, 36.9964218, -48.8242455, 37.1183739, -85.8277435, 85.8206635
17: -79.5049515, 34.1221313, -79.7975388, 34.0480194, -113.5529709, 113.9196701
18: -48.2382774, 33.2275505, -48.3332100, 33.3511086, -81.5893860, 81.5607605
19: -38.2263565, 19.1290073, -38.3477249, 19.2259521, -57.4523087, 57.4767303
20: -34.7096405, 24.9224396, -34.8271675, 24.9407082, -59.6503487, 59.7496071
21: -46.1397629, 24.6777515, -46.3583908, 24.8133545, -70.9531174, 71.0361404
22: -49.0837593, 25.2136860, -49.1267853, 25.2026558, -74.2864151, 74.3404694
23: -37.8209915, 26.2157288, -37.9159470, 26.3021049, -64.1230927, 64.1316757
24: -45.4657288, 28.8163166, -45.4781380, 28.8951874, -74.3609161, 74.2944565
25: -39.6955795, 29.4056149, -39.7334633, 29.4466286, -69.1422119, 69.1390762
26: -55.9894485, 38.6334686, -56.1085815, 38.6194191, -94.6088715, 94.7420502
27: -46.0078583, 30.0785408, -46.0574188, 30.1956272, -76.2034836, 76.1359558
28: -36.9702301, 29.7934055, -37.0085831, 29.8859978, -66.8562317, 66.8019867
29: -51.1882477, 24.5688171, -51.2416039, 24.5241470, -75.7123947, 75.8104248
30: -46.3201256, 33.2405243, -46.3803062, 33.3639069, -79.6840363, 79.6208344
31: -49.0815659, 27.6414852, -49.2045670, 27.7226982, -76.8042603, 76.8460541
32: -55.5631180, 24.5645275, -55.7137871, 24.5750027, -80.0106201, 80.1682892
33: -73.6892395, 31.7487278, -73.7232666, 32.0055008, -105.2089157, 104.8268127
34: -63.6422577, 17.8162537, -63.7029419, 17.9968376, -81.3936005, 81.0395660
35: -60.7080688, 24.2963028, -60.7863503, 24.5559978, -84.7316437, 84.4111099
36: -60.8149071, 25.3980103, -60.8648987, 25.3574810, -86.1713867, 86.2622223
37: -89.4443283, 18.4875870, -89.4939117, 18.6251736, -107.8979950, 107.8147812
38: -69.6938324, 29.1535931, -69.7895355, 29.1205158, -98.8143463, 98.9431305
39: -83.3546448, 30.7131233, -83.4347382, 30.8684387, -114.2230835, 114.1478577
40: -65.8100739, 21.4296246, -65.8610077, 21.4943619, -87.2962952, 87.2502441
41: -58.7450409, 28.5768280, -58.7955627, 28.6316395, -87.3766785, 87.3723907
42: -40.1609573, 24.5087490, -40.3882599, 24.5955296, -64.7564850, 64.8970108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1739169
time: 80.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1820517
time: 103.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -57.3780136, 42.9693108, -56.9939728, 42.8899460, -100.2679596, 99.9632874
1: -26.6460037, 35.3023720, -26.3415909, 35.2622185, -61.9082222, 61.6439629
2: -24.5326538, 37.0565643, -24.0709553, 37.0614433, -61.5940971, 61.1275177
3: -28.8138313, 41.4545822, -28.3045349, 41.3746643, -70.1884918, 69.7591171
4: -31.7397614, 41.3051224, -31.2550240, 41.3228989, -73.0626602, 72.5601501
5: -28.1476936, 42.7334175, -27.6819725, 42.6662521, -70.8139496, 70.4153900
6: -55.2889824, 27.2576046, -55.2496223, 27.0706158, -82.3595963, 82.5072250
7: -32.6788292, 40.5661087, -32.1802368, 40.4910583, -73.1698914, 72.7463455
8: -37.2721672, 49.4459534, -36.8372345, 49.4398613, -86.7120285, 86.2831879
9: -30.3847580, 38.4018707, -30.3248005, 38.1197052, -68.5044632, 68.7266693
10: -49.6579819, 48.3149719, -49.7761230, 47.6150513, -97.2730331, 98.0910950
11: -48.5813370, 29.0708046, -48.7096596, 28.6757202, -77.2570572, 77.7804642
12: -59.9048233, 31.8194466, -60.0033112, 30.7892952, -90.2619324, 91.4205933
13: -51.3782959, 47.2074966, -51.2704086, 46.9051552, -98.2834473, 98.4779053
14: -79.6990051, 43.0064087, -79.6407242, 42.2367020, -121.9357071, 122.6471329
15: -38.1694374, 35.2501831, -37.8605652, 35.2399673, -73.4094086, 73.1107483
16: -48.8793602, 37.2686119, -48.7030716, 36.9477692, -85.8271332, 85.9716797
17: -79.6598511, 34.4753799, -79.6641388, 33.8101730, -113.4700241, 114.1395187
18: -48.3746796, 33.4516983, -48.2312737, 33.2015762, -81.5762558, 81.6829681
19: -38.3363342, 19.2536545, -38.2648849, 19.1438999, -57.4802322, 57.5185394
20: -34.8308487, 25.0604877, -34.7364044, 24.8484497, -59.6792984, 59.7968903
21: -46.2798462, 24.9052067, -46.2468529, 24.6608925, -70.9407349, 71.1520615
22: -49.2038231, 25.3829727, -49.0401726, 25.0899658, -74.2937927, 74.4231415
23: -37.9201965, 26.3319588, -37.8396111, 26.2246723, -64.1448669, 64.1715698
24: -45.5884590, 28.8704815, -45.3937187, 28.8598652, -74.4483261, 74.2641983
25: -39.7794113, 29.5638809, -39.6737137, 29.3399544, -69.1193695, 69.2375946
26: -56.1514702, 39.0705795, -55.9759979, 38.3279190, -94.4793854, 95.0465775
27: -46.1937866, 30.1207924, -45.9312057, 30.1732941, -76.3670807, 76.0520020
28: -37.0615540, 29.8638687, -36.9447670, 29.8371906, -66.8987427, 66.8086395
29: -51.2817230, 24.7983589, -51.1756935, 24.3720188, -75.6537399, 75.9740524
30: -46.4214897, 33.4705772, -46.3028984, 33.2138252, -79.6353149, 79.7734756
31: -49.2330246, 27.7836189, -49.0933113, 27.6301193, -76.8631439, 76.8769302
32: -55.6633530, 24.7606144, -55.6406555, 24.4462261, -79.9852295, 80.2849426
33: -73.9423828, 31.8893757, -73.5514908, 31.8972721, -105.3764801, 104.7951431
34: -63.7966080, 17.9278870, -63.5981026, 17.9135151, -81.4967957, 81.0444260
35: -60.9011383, 24.4019680, -60.6588287, 24.4712582, -84.8674469, 84.3852158
36: -60.9256935, 25.4752502, -60.7871437, 25.3056622, -86.2303619, 86.2614822
37: -89.5768890, 18.6398735, -89.4014740, 18.5201569, -107.9262924, 107.8713837
38: -69.8615341, 29.2404003, -69.6734619, 29.0586472, -98.9201813, 98.9138641
39: -83.5015640, 30.7995338, -83.3319092, 30.8030663, -114.3046265, 114.1314392
40: -65.9643097, 21.4977913, -65.7531509, 21.4495583, -87.4087372, 87.2093964
41: -58.8472366, 28.6812801, -58.7270126, 28.5638618, -87.4111023, 87.4082947
42: -40.2709961, 24.7745323, -40.3001175, 24.4211140, -64.6921082, 65.0746460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0678078
time: 223.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0742994
time: 105.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -57.3780136, 42.9693108, -57.2667084, 42.9728584, -100.3508759, 100.2360229
1: -26.6460037, 35.3023720, -26.5436687, 35.3250961, -61.9710999, 61.8460388
2: -24.5326538, 37.0565643, -24.3987656, 37.1445007, -61.6771545, 61.4553299
3: -28.8138313, 41.4545822, -28.6600704, 41.5058823, -70.3197174, 70.1146545
4: -31.7397614, 41.3051224, -31.6105537, 41.4249649, -73.1647263, 72.9156799
5: -28.1476936, 42.7334175, -28.0311871, 42.7879486, -70.9356384, 70.7646027
6: -55.2889824, 27.2576046, -55.3552132, 27.2167168, -82.5056992, 82.6128159
7: -32.6788292, 40.5661087, -32.4967003, 40.5727310, -73.2515564, 73.0628052
8: -37.2721672, 49.4459534, -37.1596336, 49.5421181, -86.8142853, 86.6055908
9: -30.3847580, 38.4018707, -30.4363213, 38.3659058, -68.7506638, 68.8381958
10: -49.6579819, 48.3149719, -49.9746399, 48.1519775, -97.8099594, 98.2896118
11: -48.5813370, 29.0708046, -48.8783875, 29.0401897, -77.6215286, 77.9491882
12: -59.9048233, 31.8194466, -60.1833344, 31.3924541, -90.8713150, 91.6040802
13: -51.3782959, 47.2074966, -51.3825874, 47.0635529, -98.4418488, 98.5900879
14: -79.6990051, 43.0064087, -79.8691101, 42.6694946, -122.3684998, 122.8755188
15: -38.1694374, 35.2501831, -38.1379776, 35.3451462, -73.5145874, 73.3881607
16: -48.8793602, 37.2686119, -48.8723373, 37.2170868, -86.0964508, 86.1409454
17: -79.6598511, 34.4753799, -79.8302460, 34.1645355, -113.8243866, 114.3056259
18: -48.3746796, 33.4516983, -48.3701477, 33.4269409, -81.8016205, 81.8218460
19: -38.3363342, 19.2536545, -38.3766632, 19.2673931, -57.6037292, 57.6303177
20: -34.8308487, 25.0604877, -34.8590126, 24.9858437, -59.8166924, 59.9195023
21: -46.2798462, 24.9052067, -46.3898430, 24.8884869, -71.1683350, 71.2950516
22: -49.2038231, 25.3829727, -49.1610641, 25.2575264, -74.4613495, 74.5440369
23: -37.9201965, 26.3319588, -37.9401474, 26.3411102, -64.2613068, 64.2721100
24: -45.5884590, 28.8704815, -45.5166664, 28.9131889, -74.5016479, 74.3871460
25: -39.7794113, 29.5638809, -39.7592926, 29.4971161, -69.2765274, 69.3231735
26: -56.1514702, 39.0705795, -56.1457100, 38.7663536, -94.9178238, 95.2162933
27: -46.1937866, 30.1207924, -46.1156883, 30.2151566, -76.4089432, 76.2364807
28: -37.0615540, 29.8638687, -37.0367279, 29.9071007, -66.9686584, 66.9005966
29: -51.2817230, 24.7983589, -51.2706909, 24.6001873, -75.8819122, 76.0690460
30: -46.4214897, 33.4705772, -46.4080544, 33.4459801, -79.8674698, 79.8786316
31: -49.2330246, 27.7836189, -49.2461052, 27.7718353, -77.0048599, 77.0297241
32: -55.6633530, 24.7606144, -55.7422104, 24.6426048, -80.1817398, 80.3902283
33: -73.9423828, 31.8893757, -73.8046570, 32.0385590, -105.5106201, 105.0647278
34: -63.7966080, 17.9278870, -63.7522736, 18.0254803, -81.6055527, 81.2258530
35: -60.9011383, 24.4019680, -60.8513107, 24.5781879, -84.9717178, 84.6015625
36: -60.9256935, 25.4752502, -60.8978882, 25.3827229, -86.3075562, 86.3725433
37: -89.5768890, 18.6398735, -89.5359650, 18.6680946, -108.0805359, 108.0111542
38: -69.8615341, 29.2404003, -69.8399506, 29.1454868, -99.0070190, 99.0803528
39: -83.5015640, 30.7995338, -83.4791794, 30.8894520, -114.3910141, 114.2787170
40: -65.9643097, 21.4977913, -65.9072723, 21.5171127, -87.4786606, 87.3683929
41: -58.8472366, 28.6812801, -58.8291321, 28.6678867, -87.5151215, 87.5104141
42: -40.2709961, 24.7745323, -40.4136353, 24.6823215, -64.9533157, 65.1881714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1137751
time: 77.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1156937
time: 106.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 186.04 seconds
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1739169
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1820517
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0678078
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0742994
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1137751
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 186.04
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1156937

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.0932465, 42.8851814, -57.1733475, 42.9509811, -100.0442276, 100.0585327
1: -26.4346733, 35.2394409, -26.4753799, 35.3102798, -61.7449532, 61.7148209
2: -24.1945419, 36.9791832, -24.2868652, 37.1303024, -61.3248444, 61.2660484
3: -28.4462147, 41.3280182, -28.5372276, 41.4806404, -69.9268570, 69.8652496
4: -31.3693523, 41.2004433, -31.4916210, 41.4049911, -72.7743454, 72.6920624
5: -27.7863922, 42.6144257, -27.9122410, 42.7656174, -70.5520096, 70.5266647
6: -55.1810913, 27.0945339, -55.3197174, 27.1631203, -82.3442078, 82.4142532
7: -32.3486404, 40.4864502, -32.3907738, 40.5552139, -72.9038544, 72.8772278
8: -36.9369583, 49.3404732, -37.0520935, 49.5202179, -86.4571762, 86.3925629
9: -30.2661705, 38.1477661, -30.4028263, 38.2832489, -68.5494232, 68.5505905
10: -49.4556084, 47.7606583, -49.9335098, 47.9752655, -97.4308777, 97.6941681
11: -48.4064865, 28.6745186, -48.8429222, 28.9080181, -77.3145065, 77.5174408
12: -59.7227783, 31.1941776, -60.1558304, 31.1903076, -90.4837036, 90.9504852
13: -51.2577744, 47.0341301, -51.3408279, 47.0100555, -98.2678299, 98.3749542
14: -79.4647598, 42.5612717, -79.8132782, 42.5261002, -121.9908600, 122.3745499
15: -37.8504219, 35.1497116, -38.0249443, 35.3113174, -73.1617432, 73.1746521
16: -48.6896858, 36.9804535, -48.8191719, 37.1141815, -85.8038635, 85.7996216
17: -79.4984741, 34.1062622, -79.7958832, 34.0437050, -113.5421753, 113.9021454
18: -48.2320976, 33.2191544, -48.3315811, 33.3488922, -81.5809937, 81.5507355
19: -38.2216644, 19.1221313, -38.3464813, 19.2241001, -57.4457626, 57.4686127
20: -34.7049713, 24.9155960, -34.8259659, 24.9388733, -59.6438446, 59.7415619
21: -46.1343727, 24.6686707, -46.3569946, 24.8106613, -70.9450378, 71.0256653
22: -49.0776329, 25.2036266, -49.1252251, 25.1994019, -74.2770386, 74.3288498
23: -37.8172455, 26.2085876, -37.9149666, 26.3002243, -64.1174698, 64.1235504
24: -45.4588890, 28.8108482, -45.4763298, 28.8937321, -74.3526230, 74.2871780
25: -39.6898308, 29.3993683, -39.7319641, 29.4453526, -69.1351852, 69.1313324
26: -55.9829407, 38.6180038, -56.1069107, 38.6153107, -94.5982513, 94.7249146
27: -46.0020027, 30.0719223, -46.0558701, 30.1938457, -76.1958466, 76.1277924
28: -36.9660187, 29.7872162, -37.0074425, 29.8843384, -66.8503571, 66.7946625
29: -51.1824188, 24.5585804, -51.2400970, 24.5210190, -75.7034378, 75.7986755
30: -46.3139610, 33.2208900, -46.3786850, 33.3588333, -79.6727905, 79.5995789
31: -49.0758324, 27.6301880, -49.2030640, 27.7199097, -76.7957458, 76.8332520
32: -55.5584526, 24.5533981, -55.7125702, 24.5716667, -80.0034637, 80.1549606
33: -73.6788483, 31.7433586, -73.7205353, 32.0041351, -105.1888275, 104.8187408
34: -63.6351547, 17.8113213, -63.7010193, 17.9955196, -81.3723526, 81.0327835
35: -60.6988144, 24.2927952, -60.7839432, 24.5550804, -84.7149124, 84.4049377
36: -60.8104477, 25.3891277, -60.8637543, 25.3551273, -86.1646652, 86.2521515
37: -89.4361877, 18.4740562, -89.4917755, 18.6215820, -107.8877487, 107.7979584
38: -69.6871338, 29.1444931, -69.7877884, 29.1181412, -98.8052750, 98.9322815
39: -83.3465805, 30.6955986, -83.4325867, 30.8639603, -114.2105408, 114.1281891
40: -65.7995453, 21.4240665, -65.8582764, 21.4929142, -87.2837677, 87.2416687
41: -58.7399330, 28.5600624, -58.7942085, 28.6272888, -87.3672180, 87.3542709
42: -40.1568298, 24.4976330, -40.3872032, 24.5925255, -64.7493591, 64.8848343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=246, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0223077, upper bound: 38.1800432
time: 87.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9114220, upper bound: 38.0665542
time: 225.08 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 314.89 seconds
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 314.89
Output dim: 2, lower bound: -38.0223077, upper bound: 38.1800432
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 314.89
Output dim: 2, lower bound: -37.9114220, upper bound: 38.0665542

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 161.94 + 2937.45 = 3099.39 seconds
