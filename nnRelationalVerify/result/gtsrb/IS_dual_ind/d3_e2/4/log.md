## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.83 + 162.02 = 164.85 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2120267, upper bound: 38.1481822
time: 134.13 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2120267, upper bound: 38.2120264
time: 102.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 237.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 237.11
Output dim: 2, lower bound: -38.2120267, upper bound: 38.1481822
IS_A2, status: Status.UNKNOWN, split count: 1, time: 237.11
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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087034, upper bound: 38.0807511
time: 108.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2088659, upper bound: 38.1449489
time: 94.04 seconds

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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087034, upper bound: 38.1446980
time: 139.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2088659, upper bound: 38.2088653
time: 71.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 213.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 213.78
Output dim: 2, lower bound: -38.2087034, upper bound: 38.0807511
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 213.78
Output dim: 2, lower bound: -38.2088659, upper bound: 38.1449489
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 213.78
Output dim: 2, lower bound: -38.2087034, upper bound: 38.1446980
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 213.78
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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
time: 111.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
time: 89.27 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1253291
time: 96.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1424323
time: 121.28 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1210255
time: 76.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1394054
time: 86.49 seconds

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

Time for backsubstitution: 2.26 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1882480
time: 88.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0770229, upper bound: 38.2063348
time: 98.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 189.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.0578789
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1253291
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1424323
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1210255
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1394054
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.1882480
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 189.54
Output dim: 2, lower bound: -38.0770229, upper bound: 38.2063348

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -56.8581772, 42.7077751, -57.0049438, 42.8244438, -99.6826172, 99.7127228
1: -26.2535248, 35.0760040, -26.3690910, 35.1886635, -61.4421883, 61.4450951
2: -23.9713173, 36.8331299, -24.1101074, 36.9547882, -60.9261055, 60.9432373
3: -28.2342453, 41.1429596, -28.3993797, 41.2910423, -69.5252838, 69.5423431
4: -31.1643887, 41.0415649, -31.3046627, 41.1755905, -72.3399811, 72.3462296
5: -27.6000767, 42.4424515, -27.7701111, 42.5801620, -70.1802368, 70.2125626
6: -55.0891800, 26.9910736, -55.1359024, 27.0534534, -82.1426315, 82.1269760
7: -32.0409698, 40.2294579, -32.2189636, 40.4082184, -72.4491882, 72.4484253
8: -36.7332916, 49.1484680, -36.8562317, 49.2894211, -86.0227127, 86.0046997
9: -30.1497879, 37.9861069, -30.2091789, 38.0689735, -68.2187653, 68.1952820
10: -49.2998657, 47.4295654, -49.4595757, 47.5831947, -96.8830566, 96.8891449
11: -48.2899437, 28.5650578, -48.4361076, 28.6875629, -76.9775085, 77.0011673
12: -59.3240585, 30.6007824, -59.6663589, 30.7988853, -89.7014160, 89.8458099
13: -51.0535698, 46.7449417, -51.2035065, 46.7716751, -97.8252411, 97.9484482
14: -79.0787735, 42.1036224, -79.3820038, 42.1886902, -121.2674637, 121.4856262
15: -37.7071686, 35.0115891, -37.8171921, 35.0646667, -72.7718353, 72.8287811
16: -48.3825912, 36.6838455, -48.4874458, 36.8581619, -85.2407532, 85.1712952
17: -79.0654984, 33.6595383, -79.3902512, 33.7469177, -112.8124161, 113.0497894
18: -48.0222740, 33.1344299, -48.1219406, 33.2541733, -81.2764435, 81.2563705
19: -38.1251831, 19.1200180, -38.1969757, 19.1954174, -57.3206024, 57.3169937
20: -34.5473251, 24.8009987, -34.6574707, 24.8557739, -59.4030991, 59.4584694
21: -46.0087929, 24.6179123, -46.1190300, 24.7350330, -70.7438278, 70.7369385
22: -48.7801933, 24.9547005, -48.9189377, 25.0485992, -73.8287964, 73.8736420
23: -37.6940918, 26.2016830, -37.7778168, 26.2706566, -63.9647484, 63.9794998
24: -45.2600594, 28.7396927, -45.2708893, 28.7942772, -74.0543365, 74.0105820
25: -39.5118294, 29.2537308, -39.5961990, 29.3485718, -68.8603973, 68.8499298
26: -55.6013985, 38.2352028, -55.8625984, 38.4507179, -94.0521164, 94.0978012
27: -45.7689323, 29.9781685, -45.7879181, 30.0429192, -75.8118515, 75.7660828
28: -36.8462791, 29.7559776, -36.8845520, 29.7900562, -66.6363373, 66.6405334
29: -50.9076920, 24.2969227, -51.0514755, 24.4113655, -75.3190613, 75.3483963
30: -46.1845894, 33.0969009, -46.2454224, 33.2397079, -79.4243011, 79.3423233
31: -48.9235115, 27.5874100, -49.0057030, 27.6775036, -76.6010132, 76.5931091
32: -55.3961296, 24.3567047, -55.4979324, 24.4126492, -79.6836090, 79.7328415
33: -73.4191284, 31.5652313, -73.4832458, 31.7173557, -104.5344162, 104.4138565
34: -63.5058365, 17.6948910, -63.5654030, 17.7931080, -80.8673935, 80.7798309
35: -60.5476265, 24.1977196, -60.6018295, 24.2999058, -84.2029495, 84.1338654
36: -60.5891418, 25.1822433, -60.7059441, 25.2329979, -85.8205261, 85.8866425
37: -89.2324829, 18.4069424, -89.2835846, 18.5072365, -107.5626068, 107.5100021
38: -69.4065704, 28.9303341, -69.5859909, 28.9891739, -98.3957443, 98.5163269
39: -83.2015076, 30.5788040, -83.2493591, 30.6940937, -113.8955994, 113.8281631
40: -65.6091461, 21.2544937, -65.6565018, 21.3965225, -86.9696350, 86.8681183
41: -58.6134682, 28.4647865, -58.6557426, 28.5657444, -87.1792145, 87.1205292
42: -40.0349579, 24.3411846, -40.1316299, 24.4365768, -64.4715347, 64.4728165

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 765
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 705
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
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
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 737
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
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
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1699
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 37.9451788
time: 84.07 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0547599
time: 108.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -57.1265297, 42.7889938, -57.0915031, 42.8448334, -99.9713593, 99.8804932
1: -26.4561634, 35.1368332, -26.4346485, 35.2028618, -61.6590271, 61.5714798
2: -24.2974358, 36.9079285, -24.2161999, 36.9686623, -61.2660980, 61.1241302
3: -28.5846634, 41.2655411, -28.5135574, 41.3151627, -69.8998260, 69.7790985
4: -31.5207596, 41.1416130, -31.4201431, 41.1948166, -72.7155762, 72.5617523
5: -27.9455566, 42.5577469, -27.8813553, 42.6018410, -70.5473938, 70.4391022
6: -55.1902924, 27.1306877, -55.1708069, 27.1007767, -82.2910690, 82.3014984
7: -32.3579979, 40.3066101, -32.3199959, 40.4249115, -72.7829132, 72.6266022
8: -37.0558548, 49.2503052, -36.9602737, 49.3106842, -86.3665390, 86.2105789
9: -30.2595081, 38.2328720, -30.2410603, 38.1477661, -68.4072723, 68.4739304
10: -49.4983215, 47.9691162, -49.5000153, 47.7594070, -97.2577286, 97.4691315
11: -48.4572563, 28.9321957, -48.4693871, 28.8076134, -77.2648697, 77.4015808
12: -59.5017204, 31.2048264, -59.6927643, 30.9949341, -90.0805817, 90.4797516
13: -51.1668205, 46.8994980, -51.2426147, 46.8190002, -97.9858246, 98.1421127
14: -79.3050766, 42.5360680, -79.4351273, 42.3293610, -121.6344376, 121.9711914
15: -37.9841118, 35.1056900, -37.9074707, 35.0973511, -73.0814667, 73.0131607
16: -48.5506821, 36.9538651, -48.5360985, 36.9471855, -85.4978638, 85.4899597
17: -79.2204742, 34.0072212, -79.4224091, 33.8577271, -113.0782013, 113.4296265
18: -48.1580353, 33.3552399, -48.1580658, 33.3259277, -81.4839630, 81.5133057
19: -38.2348862, 19.2433510, -38.2258759, 19.2356644, -57.4705505, 57.4692268
20: -34.6687546, 24.9383049, -34.6889420, 24.9008904, -59.5696449, 59.6272469
21: -46.1477509, 24.8450775, -46.1505051, 24.8086109, -70.9563599, 70.9955826
22: -48.8995590, 25.1218758, -48.9560127, 25.1018620, -74.0014191, 74.0778885
23: -37.7925873, 26.3156090, -37.8020172, 26.3082161, -64.1007996, 64.1176300
24: -45.3809967, 28.7927036, -45.3096275, 28.8125191, -74.1935120, 74.1023331
25: -39.5958214, 29.4111748, -39.6216660, 29.3985138, -68.9943390, 69.0328369
26: -55.7642403, 38.6676559, -55.8989029, 38.5908432, -94.3550873, 94.5665588
27: -45.9524155, 30.0195980, -45.8462753, 30.0558071, -76.0082245, 75.8658752
28: -36.9365845, 29.8253632, -36.9123306, 29.8118992, -66.7484818, 66.7376938
29: -51.0010071, 24.5241566, -51.0800476, 24.4857197, -75.4867249, 75.6042023
30: -46.2854652, 33.3206329, -46.2727318, 33.3113480, -79.5968170, 79.5933685
31: -49.0738564, 27.7291870, -49.0468597, 27.7241936, -76.7980499, 76.7760468
32: -55.4961319, 24.5513802, -55.5267830, 24.4772453, -79.8519592, 79.9559174
33: -73.6709518, 31.7064629, -73.5648117, 31.7504864, -104.8349380, 104.6425705
34: -63.6590195, 17.8068542, -63.6148758, 17.8217545, -81.0771484, 80.9583282
35: -60.7395973, 24.3038673, -60.6646957, 24.3221893, -84.4423065, 84.3170319
36: -60.6986732, 25.2587891, -60.7411346, 25.2547531, -85.9522781, 85.9987488
37: -89.3639374, 18.5585041, -89.3250885, 18.5554466, -107.7491684, 107.7101135
38: -69.5729294, 29.0164948, -69.6388474, 29.0137253, -98.5866547, 98.6553421
39: -83.3474503, 30.6652451, -83.2944260, 30.7153187, -114.0627670, 113.9596710
40: -65.7612686, 21.3217659, -65.7027359, 21.4176788, -87.1485291, 86.9862747
41: -58.7144661, 28.5687962, -58.6896248, 28.5992088, -87.3136749, 87.2584229
42: -40.1436119, 24.5923157, -40.1574707, 24.5200386, -64.6636505, 64.7497864

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
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
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 37.9765738
time: 101.48 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 37.9765738
time: 115.02 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -56.8846512, 42.7208252, -57.1726608, 42.9195557, -99.8042068, 99.8934860
1: -26.2715206, 35.0850258, -26.4769459, 35.2782021, -61.5497208, 61.5619736
2: -24.0147438, 36.8409729, -24.2929001, 37.1014862, -61.1162300, 61.1338730
3: -28.2653980, 41.1588097, -28.5464592, 41.4499397, -69.7153397, 69.7052689
4: -31.2069359, 41.0531044, -31.5007019, 41.3766289, -72.5835648, 72.5538025
5: -27.6291904, 42.4585648, -27.9190903, 42.7375336, -70.3667221, 70.3776550
6: -55.1086807, 27.0124245, -55.3166008, 27.1766014, -82.2852783, 82.3290253
7: -32.0707779, 40.2410774, -32.3938293, 40.5007095, -72.5714874, 72.6349030
8: -36.7808228, 49.1628265, -37.0607758, 49.4865913, -86.2674103, 86.2236023
9: -30.1643772, 38.0383301, -30.4042244, 38.2823792, -68.4467545, 68.4425507
10: -49.3229904, 47.5362320, -49.9165535, 47.9775200, -97.3005066, 97.4527893
11: -48.3109016, 28.6340065, -48.8290443, 28.9399185, -77.2508240, 77.4630508
12: -59.3401794, 30.7065430, -60.0692101, 31.1997490, -90.1171951, 90.3589172
13: -51.0659523, 46.7933350, -51.3209457, 47.0071182, -98.0730743, 98.1142807
14: -79.1016235, 42.1991539, -79.7492218, 42.5336075, -121.6352310, 121.9483795
15: -37.7625923, 35.0299416, -38.0635223, 35.3050079, -73.0675964, 73.0934601
16: -48.4088211, 36.7457733, -48.8140907, 37.0949020, -85.5037231, 85.5598602
17: -79.0798645, 33.7307968, -79.7073822, 34.0519714, -113.1318359, 113.4381790
18: -48.0459671, 33.1489105, -48.3117752, 33.3541718, -81.4001389, 81.4606857
19: -38.1427002, 19.1262665, -38.3368988, 19.2440376, -57.3867378, 57.4631653
20: -34.5643387, 24.8219604, -34.8085251, 24.9415970, -59.5059357, 59.6304855
21: -46.0277672, 24.6390629, -46.3443031, 24.8321915, -70.8599548, 70.9833679
22: -48.8076286, 24.9735775, -49.0778236, 25.2056904, -74.0133209, 74.0513992
23: -37.7088089, 26.2065849, -37.9016876, 26.3171997, -64.0260086, 64.1082764
24: -45.3067436, 28.7459793, -45.4717789, 28.8912735, -74.1980133, 74.2177582
25: -39.5314102, 29.2669907, -39.7076416, 29.4454117, -68.9768219, 68.9746323
26: -55.6232567, 38.2704544, -56.0392723, 38.6258011, -94.2490540, 94.3097229
27: -45.8305511, 29.9845905, -46.0500717, 30.1842728, -76.0148239, 76.0346603
28: -36.8687057, 29.7646027, -36.9995346, 29.8946953, -66.7633972, 66.7641373
29: -50.9291077, 24.3184586, -51.1924133, 24.5254822, -75.4545898, 75.5108719
30: -46.2028275, 33.1195602, -46.3749237, 33.3662643, -79.5690918, 79.4944839
31: -48.9498405, 27.5965214, -49.1938248, 27.7359810, -76.6858215, 76.7903442
32: -55.4148026, 24.3984547, -55.6896362, 24.5758286, -79.8652344, 79.9702530
33: -73.4788437, 31.5812721, -73.7190094, 31.9796658, -104.8914108, 104.6749039
34: -63.5355225, 17.7077637, -63.6992798, 17.9829712, -81.1386719, 80.9412231
35: -60.5930977, 24.2102928, -60.7831612, 24.5427055, -84.5272751, 84.3349152
36: -60.6126518, 25.1926041, -60.8286743, 25.3560791, -85.9676285, 86.0199127
37: -89.2682343, 18.4213047, -89.4798126, 18.6272907, -107.7247772, 107.7237778
38: -69.4304047, 28.9441395, -69.7454376, 29.1150379, -98.5454407, 98.6895752
39: -83.2321243, 30.5878429, -83.4288635, 30.8471050, -114.0792313, 114.0167084
40: -65.6448059, 21.2591324, -65.8589783, 21.4647903, -87.0808105, 87.0803528
41: -58.6339912, 28.4736557, -58.7908363, 28.6286736, -87.2626648, 87.2644958
42: -40.0510330, 24.3659210, -40.3719330, 24.5951614, -64.6461945, 64.7378540

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 761
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0135069
time: 94.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1221865
time: 110.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.1560478, 42.8019333, -57.2622986, 42.9400482, -100.0960999, 100.0642319
1: -26.4744949, 35.1458282, -26.5423336, 35.2923050, -61.7667999, 61.6881638
2: -24.3409119, 36.9158707, -24.3991909, 37.1152229, -61.4561348, 61.3150635
3: -28.6189480, 41.2813873, -28.6631069, 41.4740601, -70.0930099, 69.9444962
4: -31.5634422, 41.1540947, -31.6159630, 41.3959198, -72.9593658, 72.7700577
5: -27.9775429, 42.5737381, -28.0322762, 42.7589493, -70.7364960, 70.6060181
6: -55.2106209, 27.1555290, -55.3510818, 27.2255077, -82.4361267, 82.5066071
7: -32.3881912, 40.3182755, -32.4947510, 40.5173721, -72.9055634, 72.8130264
8: -37.1039581, 49.2645721, -37.1646957, 49.5075035, -86.6114655, 86.4292679
9: -30.2744064, 38.2856064, -30.4358501, 38.3610764, -68.6354828, 68.7214584
10: -49.5214691, 48.0761070, -49.9565849, 48.1532860, -97.6747589, 98.0326920
11: -48.4781914, 29.0017586, -48.8621407, 29.0599232, -77.5381165, 77.8638992
12: -59.5184212, 31.3106728, -60.0956192, 31.3956108, -90.4966736, 90.9928436
13: -51.1794434, 46.9485588, -51.3598976, 47.0555420, -98.2349854, 98.3084564
14: -79.3284149, 42.6317863, -79.8024673, 42.6745148, -122.0029297, 122.4342499
15: -38.0432243, 35.1242218, -38.1557999, 35.3372154, -73.3804398, 73.2800217
16: -48.5774117, 37.0162773, -48.8624611, 37.1840401, -85.7614517, 85.8787384
17: -79.2351913, 34.0827446, -79.7396545, 34.1657372, -113.4009247, 113.8224030
18: -48.1817970, 33.3723907, -48.3487854, 33.4282150, -81.6100159, 81.7211761
19: -38.2522888, 19.2502956, -38.3660431, 19.2840557, -57.5363464, 57.6163406
20: -34.6858063, 24.9594536, -34.8398285, 24.9867096, -59.6725159, 59.7992821
21: -46.1672249, 24.8661461, -46.3757553, 24.9054718, -71.0726929, 71.2418976
22: -48.9276733, 25.1414948, -49.1153488, 25.2588081, -74.1864777, 74.2568436
23: -37.8077011, 26.3222179, -37.9260178, 26.3560867, -64.1637878, 64.2482376
24: -45.4281311, 28.7999344, -45.5110741, 28.9094486, -74.3375778, 74.3110046
25: -39.6156387, 29.4245300, -39.7335587, 29.4951706, -69.1108093, 69.1580887
26: -55.7863121, 38.7066345, -56.0758438, 38.7689056, -94.5552216, 94.7824783
27: -46.0144768, 30.0263462, -46.1088448, 30.1971016, -76.2115784, 76.1351929
28: -36.9593086, 29.8344421, -37.0277939, 29.9162712, -66.8755798, 66.8622360
29: -51.0225983, 24.5470028, -51.2213097, 24.6007729, -75.6233673, 75.7683105
30: -46.3037415, 33.3491135, -46.4021912, 33.4426117, -79.7463531, 79.7513046
31: -49.0996742, 27.7390900, -49.2353439, 27.7826614, -76.8823395, 76.9744339
32: -55.5151176, 24.5934963, -55.7183418, 24.6414604, -80.0344238, 80.1934586
33: -73.7312164, 31.7223740, -73.8008118, 32.0124741, -105.1920319, 104.9035797
34: -63.6891670, 17.8196468, -63.7489624, 18.0114098, -81.3485489, 81.1197968
35: -60.7853699, 24.3161297, -60.8461838, 24.5647697, -84.7664795, 84.5178604
36: -60.7234612, 25.2691460, -60.8649597, 25.3776379, -86.1004028, 86.1330719
37: -89.4003906, 18.5728912, -89.5220566, 18.6740971, -107.9103394, 107.9250641
38: -69.5979919, 29.0303211, -69.7985535, 29.1395588, -98.7375488, 98.8288727
39: -83.3783569, 30.6743183, -83.4741669, 30.8681850, -114.2465439, 114.1484833
40: -65.7979584, 21.3264389, -65.9057159, 21.4859161, -87.2604446, 87.1990662
41: -58.7355118, 28.5777321, -58.8247299, 28.6621914, -87.3977051, 87.4024658
42: -40.1606522, 24.6290760, -40.3976746, 24.6838169, -64.8444672, 65.0267487

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
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
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 552
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

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0138102
time: 82.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0138102
time: 181.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -57.0794487, 42.8755760, -57.0160484, 42.8585510, -99.9380035, 99.8916245
1: -26.4251270, 35.2328720, -26.3759575, 35.2224998, -61.6476288, 61.6088295
2: -24.1623077, 36.9741325, -24.1154633, 36.9856377, -61.1479454, 61.0895958
3: -28.4284821, 41.3167992, -28.4031372, 41.3253632, -69.7538452, 69.7199402
4: -31.3395386, 41.1923828, -31.3074417, 41.2069130, -72.5464478, 72.4998245
5: -27.7693958, 42.6026955, -27.7756710, 42.6114426, -70.3808365, 70.3783646
6: -55.1672211, 27.0914917, -55.1433640, 27.0584412, -82.2256622, 82.2348557
7: -32.3311234, 40.4778900, -32.2283173, 40.4651604, -72.7962799, 72.7062073
8: -36.9006042, 49.3302383, -36.8584785, 49.3261223, -86.2267303, 86.1887207
9: -30.2592621, 38.1026535, -30.2129288, 38.0785522, -68.3378143, 68.3155823
10: -49.4382553, 47.6671143, -49.4813156, 47.5955048, -97.0337601, 97.1484299
11: -48.3923683, 28.6363792, -48.4552040, 28.6783504, -77.0707169, 77.0915833
12: -59.7113991, 31.1083851, -59.7565842, 30.8096390, -90.0911179, 90.4561310
13: -51.2529602, 47.0023422, -51.2385941, 46.7835541, -98.0365143, 98.2409363
14: -79.4503937, 42.4775162, -79.4535065, 42.1936417, -121.6440353, 121.9310226
15: -37.8370132, 35.1358719, -37.8141251, 35.0751266, -72.9121399, 72.9499969
16: -48.6830368, 36.9357605, -48.5023804, 36.9006729, -85.5837097, 85.4381409
17: -79.4908142, 34.0511475, -79.4833984, 33.7522430, -113.2430573, 113.5345459
18: -48.2147560, 33.2133560, -48.1465988, 33.2575874, -81.4723434, 81.3599548
19: -38.2090149, 19.1229553, -38.2102318, 19.1812572, -57.3902740, 57.3331871
20: -34.6928902, 24.9016571, -34.6789246, 24.8587265, -59.5516167, 59.5805817
21: -46.1208725, 24.6569290, -46.1360207, 24.7217999, -70.8426743, 70.7929535
22: -49.0563431, 25.1950531, -48.9713936, 25.0546722, -74.1110153, 74.1664429
23: -37.8063431, 26.2112503, -37.7942963, 26.2618694, -64.0682144, 64.0055466
24: -45.4192085, 28.8103981, -45.2832413, 28.8036213, -74.2228317, 74.0936432
25: -39.6763535, 29.3924980, -39.6251907, 29.3542175, -69.0305710, 69.0176849
26: -55.9678383, 38.5984306, -55.9357529, 38.4560661, -94.4239044, 94.5341797
27: -45.9462433, 30.0725822, -45.8000565, 30.0623837, -76.0086288, 75.8726349
28: -36.9478455, 29.7848892, -36.8960953, 29.7849388, -66.7327881, 66.6809845
29: -51.1671104, 24.5474358, -51.1043434, 24.4154568, -75.5825653, 75.6517792
30: -46.3019791, 33.2182999, -46.2539825, 33.2492790, -79.5512543, 79.4722824
31: -49.0550690, 27.6328850, -49.0199089, 27.6716805, -76.7267456, 76.6527939
32: -55.5446625, 24.5229988, -55.5248260, 24.4177761, -79.8353424, 79.9308014
33: -73.6297607, 31.7328815, -73.4938202, 31.7461052, -104.8550262, 104.5886841
34: -63.6127129, 17.8035431, -63.5730896, 17.8096142, -81.1261902, 80.8998184
35: -60.6629333, 24.2838955, -60.6125336, 24.3148441, -84.4093704, 84.2300949
36: -60.7914124, 25.3878555, -60.7447395, 25.2394257, -86.0291748, 86.1315460
37: -89.4087830, 18.4734612, -89.3023453, 18.5091991, -107.7406998, 107.6000519
38: -69.6704025, 29.1399956, -69.6343460, 28.9987087, -98.6691132, 98.7743378
39: -83.3242111, 30.7041492, -83.2594757, 30.7172050, -114.0414124, 113.9636230
40: -65.7744751, 21.4255829, -65.6624756, 21.4341927, -87.1939545, 87.0426712
41: -58.7245865, 28.5685959, -58.6632385, 28.5795250, -87.3041077, 87.2318344
42: -40.1450348, 24.4845314, -40.1501770, 24.4443150, -64.5893478, 64.6347046

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 761
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
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 765
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 679
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
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 705
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
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
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 737
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
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

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0042115
time: 110.98 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1178783
time: 85.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -57.3484917, 42.9565430, -57.1025887, 42.8789825, -100.2274780, 100.0591278
1: -26.6275597, 35.2934380, -26.4415188, 35.2367096, -61.8642693, 61.7349548
2: -24.4891663, 37.0486946, -24.2215385, 36.9994926, -61.4886589, 61.2702332
3: -28.7794418, 41.4388657, -28.5173149, 41.3494949, -70.1289368, 69.9561768
4: -31.6969833, 41.2926750, -31.4229279, 41.2261429, -72.9231262, 72.7156067
5: -28.1156006, 42.7175446, -27.8868256, 42.6331177, -70.7487183, 70.6043701
6: -55.2686234, 27.2326241, -55.1782913, 27.1059685, -82.3745880, 82.4109192
7: -32.6485748, 40.5545502, -32.3293495, 40.4818573, -73.1304321, 72.8838959
8: -37.2240105, 49.4317780, -36.9625511, 49.3473663, -86.5713806, 86.3943329
9: -30.3696594, 38.3492241, -30.2448521, 38.1573563, -68.5270157, 68.5940781
10: -49.6351814, 48.2078323, -49.5215263, 47.7716370, -97.4068146, 97.7293549
11: -48.5603104, 29.0014858, -48.4885635, 28.7983322, -77.3586426, 77.4900513
12: -59.8882751, 31.7134399, -59.7829590, 31.0056839, -90.4694519, 91.0910339
13: -51.3656998, 47.1582146, -51.2776642, 46.8309517, -98.1966553, 98.4358826
14: -79.6757050, 42.9105453, -79.5065994, 42.3343277, -122.0100327, 122.4171448
15: -38.1107216, 35.2311821, -37.9039688, 35.1078720, -73.2185974, 73.1351471
16: -48.8522377, 37.2063026, -48.5511398, 36.9896507, -85.8418884, 85.7574463
17: -79.6452637, 34.3996658, -79.5155716, 33.8630791, -113.5083466, 113.9152374
18: -48.3509331, 33.4344902, -48.1826897, 33.3293686, -81.6802979, 81.6171799
19: -38.3189926, 19.2466869, -38.2391968, 19.2216148, -57.5406075, 57.4858856
20: -34.8138695, 25.0393028, -34.7103806, 24.9038258, -59.7176971, 59.7496834
21: -46.2603226, 24.8841476, -46.1675911, 24.7953377, -71.0556641, 71.0517426
22: -49.1757812, 25.3630943, -49.0084267, 25.1078434, -74.2836227, 74.3715210
23: -37.9051514, 26.3253422, -37.8185463, 26.2994480, -64.2045975, 64.1438904
24: -45.5410843, 28.8632832, -45.3219299, 28.8218479, -74.3629303, 74.1852112
25: -39.7597961, 29.5504341, -39.6505089, 29.4041634, -69.1639557, 69.2009430
26: -56.1295738, 39.0313721, -55.9720459, 38.5962372, -94.7258148, 95.0034180
27: -46.1314774, 30.1140690, -45.8584061, 30.0753117, -76.2067871, 75.9724731
28: -37.0387268, 29.8547726, -36.9238892, 29.8068542, -66.8455811, 66.7786636
29: -51.2602043, 24.7753887, -51.1329193, 24.4897938, -75.7500000, 75.9083099
30: -46.4031906, 33.4421387, -46.2813339, 33.3208084, -79.7239990, 79.7234726
31: -49.2070427, 27.7738228, -49.0609932, 27.7183762, -76.9254150, 76.8348160
32: -55.6444092, 24.7183838, -55.5536652, 24.4823532, -80.0032806, 80.1546478
33: -73.8820496, 31.8735275, -73.5753937, 31.7792263, -105.1559906, 104.8166504
34: -63.7663689, 17.9151230, -63.6226234, 17.8383179, -81.3366699, 81.0776901
35: -60.8552742, 24.3897648, -60.6753922, 24.3371830, -84.6490936, 84.4129639
36: -60.9009781, 25.4647846, -60.7798920, 25.2613220, -86.1609039, 86.2439728
37: -89.5405045, 18.6255589, -89.3439484, 18.5574684, -107.9285507, 107.8005524
38: -69.8366547, 29.2264977, -69.6871338, 29.0233288, -98.8599854, 98.9136353
39: -83.4706955, 30.7904606, -83.3045273, 30.7382908, -114.2089844, 114.0949860
40: -65.9274139, 21.4932652, -65.7087708, 21.4554806, -87.3735962, 87.1609268
41: -58.8261414, 28.6722794, -58.6971245, 28.6129837, -87.4391251, 87.3694000
42: -40.2539520, 24.7376537, -40.1760674, 24.5276985, -64.7816467, 64.9137192

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1721
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 679
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
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
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
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 704
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0397583
time: 129.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.1394054
time: 111.16 seconds

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

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0723830
time: 81.03 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1851301
time: 78.99 seconds

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

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0770224
time: 194.02 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1882486, upper bound: 38.2063348
time: 148.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 345.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 37.9451788
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0547599
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 37.9765738
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 37.9765738
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0135069
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1221865
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0138102
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0138102
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0042115
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1178783
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0397583
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.1394054
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.0723830
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.0678082, upper bound: 38.1851301
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.0770224
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 345.21
Output dim: 2, lower bound: -38.1882486, upper bound: 38.2063348

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -56.7832527, 42.6912079, -56.7774124, 42.7457504, -99.5290070, 99.4686203
1: -26.2002468, 35.0647469, -26.2082386, 35.1309357, -61.3311844, 61.2729874
2: -23.8813095, 36.8217010, -23.8404160, 36.8704453, -60.7517548, 60.6621170
3: -28.1341286, 41.1235046, -28.0993710, 41.1777916, -69.3119202, 69.2228775
4: -31.0722809, 41.0263290, -31.0278187, 41.0784149, -72.1506958, 72.0541458
5: -27.5031090, 42.4247589, -27.4781876, 42.4651680, -69.9682770, 69.9029465
6: -55.0635986, 26.9480095, -55.0531845, 26.9196854, -81.9832840, 82.0011902
7: -31.9570427, 40.2158966, -31.9622288, 40.3323593, -72.2893982, 72.1781235
8: -36.6489639, 49.1323929, -36.6009598, 49.1953430, -85.8443069, 85.7333527
9: -30.1227131, 37.9326324, -30.1236229, 37.8998413, -68.0225525, 68.0562592
10: -49.2693214, 47.3137398, -49.3172226, 47.2305870, -96.4999084, 96.6309662
11: -48.2630730, 28.4744930, -48.2940292, 28.4239960, -76.6870728, 76.7685242
12: -59.3024750, 30.4334984, -59.4975357, 30.2976322, -89.1758728, 89.5039368
13: -51.0203209, 46.7061768, -51.1013870, 46.6444168, -97.6647339, 97.8075638
14: -79.0341492, 41.9939156, -79.1919403, 41.8599701, -120.8941193, 121.1858521
15: -37.6387215, 34.9862366, -37.6098328, 34.9902687, -72.6289902, 72.5960693
16: -48.3438416, 36.6310997, -48.3540039, 36.6943436, -85.0381851, 84.9851074
17: -79.0347900, 33.5561981, -79.2258987, 33.4402237, -112.4750137, 112.7820969
18: -47.9927139, 33.0678215, -48.0038681, 33.0521278, -81.0448456, 81.0716858
19: -38.1018372, 19.0835953, -38.0928726, 19.0855694, -57.1874084, 57.1764679
20: -34.5216942, 24.7614841, -34.5542450, 24.7379360, -59.2596283, 59.3157272
21: -45.9836082, 24.5559750, -45.9927216, 24.5486202, -70.5322266, 70.5486984
22: -48.7541313, 24.9038105, -48.8319511, 24.8873692, -73.6415024, 73.7357635
23: -37.6744537, 26.1723499, -37.6914749, 26.1793995, -63.8538513, 63.8638229
24: -45.2311707, 28.7220306, -45.1769753, 28.7388783, -73.9700470, 73.8990021
25: -39.4903641, 29.2139950, -39.5266113, 29.2259064, -68.7162704, 68.7406082
26: -55.5710831, 38.1045341, -55.7076950, 38.0628319, -93.6339111, 93.8122253
27: -45.7327576, 29.9618683, -45.6661949, 29.9928818, -75.7256393, 75.6280670
28: -36.8225861, 29.7298851, -36.7909622, 29.7104702, -66.5330582, 66.5208435
29: -50.8846817, 24.2269707, -50.9575081, 24.2073746, -75.0920563, 75.1844788
30: -46.1632462, 33.0435371, -46.1674423, 33.0756874, -79.2389374, 79.2109833
31: -48.8906708, 27.5546074, -48.8860397, 27.5780487, -76.4687195, 76.4406433
32: -55.3740768, 24.2983551, -55.4029427, 24.2369080, -79.4855804, 79.5760574
33: -73.3553925, 31.5363846, -73.2888184, 31.5974617, -104.3406601, 104.1758881
34: -63.4737282, 17.6724339, -63.4634171, 17.7123032, -80.7374573, 80.6303711
35: -60.5115051, 24.1805916, -60.4895401, 24.2320557, -84.0847473, 83.9819336
36: -60.5699120, 25.1492500, -60.6425247, 25.1300049, -85.6976624, 85.7894592
37: -89.1993408, 18.3470116, -89.1524658, 18.3230858, -107.3403473, 107.3136597
38: -69.3716278, 28.8939171, -69.4675446, 28.8709545, -98.2425842, 98.3614655
39: -83.1684952, 30.5606384, -83.1441422, 30.6231117, -113.7916107, 113.7047806
40: -65.5761490, 21.2345734, -65.5472488, 21.3285065, -86.8637466, 86.7331696
41: -58.5873032, 28.4289188, -58.5689545, 28.4568863, -87.0441895, 86.9978714
42: -40.0151176, 24.2775478, -40.0348892, 24.2391376, -64.2542572, 64.3124390

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 725
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
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 679
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 608
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 537
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 37.9351437
time: 87.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0647064, upper bound: 37.9423853
time: 95.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.8577843, 42.7076721, -56.9982681, 42.8225746, -99.6803589, 99.7059402
1: -26.2531929, 35.0759392, -26.3633003, 35.1874466, -61.4406395, 61.4392395
2: -23.9708290, 36.8330574, -24.1016712, 36.9533958, -60.9242249, 60.9347305
3: -28.2336941, 41.1428223, -28.3899002, 41.2886887, -69.5223846, 69.5327225
4: -31.1638985, 41.0414352, -31.2962952, 41.1734848, -72.3373871, 72.3377304
5: -27.5995369, 42.4423141, -27.7609844, 42.5780373, -70.1775742, 70.2033005
6: -55.0889969, 26.9902134, -55.1327705, 27.0385513, -82.1275482, 82.1229858
7: -32.0405350, 40.2293472, -32.2101250, 40.4065971, -72.4471283, 72.4394684
8: -36.7328415, 49.1483383, -36.8482475, 49.2872581, -86.0200958, 85.9965820
9: -30.1496124, 37.9857063, -30.2062206, 38.0620041, -68.2116165, 68.1919250
10: -49.2996864, 47.4289055, -49.4563828, 47.5721436, -96.8718262, 96.8852844
11: -48.2897224, 28.5641937, -48.4324799, 28.6727562, -76.9624786, 76.9966736
12: -59.3239098, 30.5999069, -59.6638794, 30.7840462, -89.6860886, 89.8486938
13: -51.0527878, 46.7446899, -51.1901970, 46.7676163, -97.8204041, 97.9348907
14: -79.0784454, 42.1030579, -79.3769073, 42.1793594, -121.2578049, 121.4799652
15: -37.7057838, 35.0114136, -37.7933807, 35.0615921, -72.7673798, 72.8047943
16: -48.3822784, 36.6827507, -48.4825668, 36.8393250, -85.2216034, 85.1653137
17: -79.0653152, 33.6590919, -79.3874664, 33.7393456, -112.8046570, 113.0465546
18: -48.0220947, 33.1340523, -48.1190186, 33.2478142, -81.2699127, 81.2530670
19: -38.1250420, 19.1198006, -38.1945496, 19.1915073, -57.3165512, 57.3143501
20: -34.5471764, 24.8007812, -34.6546707, 24.8520203, -59.3991966, 59.4554520
21: -46.0086021, 24.6175900, -46.1159592, 24.7294998, -70.7380981, 70.7335510
22: -48.7799988, 24.9541702, -48.9156189, 25.0395355, -73.8195343, 73.8697891
23: -37.6939507, 26.2012501, -37.7755508, 26.2635155, -63.9574661, 63.9767990
24: -45.2596970, 28.7393723, -45.2649536, 28.7888565, -74.0485535, 74.0043259
25: -39.5116348, 29.2534771, -39.5931396, 29.3441715, -68.8558044, 68.8466187
26: -55.6011810, 38.2346725, -55.8589172, 38.4406204, -94.0418015, 94.0935898
27: -45.7686653, 29.9776974, -45.7832146, 30.0348701, -75.8035355, 75.7609100
28: -36.8461456, 29.7557869, -36.8821869, 29.7869682, -66.6331177, 66.6379700
29: -50.9074898, 24.2966442, -51.0479965, 24.4061794, -75.3136673, 75.3446426
30: -46.1843948, 33.0961227, -46.2421265, 33.2272110, -79.4116058, 79.3382492
31: -48.9233131, 27.5869675, -49.0023689, 27.6701813, -76.5934906, 76.5893402
32: -55.3959618, 24.3563919, -55.4951477, 24.4068413, -79.6775131, 79.7327881
33: -73.4187927, 31.5650635, -73.4769592, 31.7145710, -104.5313110, 104.3911514
34: -63.5055923, 17.6947250, -63.5613785, 17.7904682, -80.8632355, 80.7580795
35: -60.5471954, 24.1976204, -60.5944328, 24.2981949, -84.2007599, 84.1139145
36: -60.5889969, 25.1819592, -60.7036514, 25.2280025, -85.8155212, 85.8842010
37: -89.2322006, 18.4067020, -89.2791901, 18.5030422, -107.5565338, 107.5111084
38: -69.4063263, 28.9300575, -69.5818558, 28.9850788, -98.3914032, 98.5119171
39: -83.2012482, 30.5786858, -83.2452164, 30.6922340, -113.8934784, 113.8238983
40: -65.6089172, 21.2540131, -65.6527252, 21.3884010, -86.9606323, 86.8638611
41: -58.6133003, 28.4641495, -58.6528969, 28.5549660, -87.1682663, 87.1170502
42: -40.0348053, 24.3405724, -40.1292419, 24.4296074, -64.4644165, 64.4698181

Time for backsubstitution: 2.20 seconds

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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 725
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
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 679
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 608
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 606
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
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0438518
time: 92.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0517390
time: 107.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -57.1265297, 42.7889938, -56.8185883, 42.7622643, -99.8887939, 99.6075821
1: -26.4561634, 35.1368332, -26.2266808, 35.1405792, -61.5967407, 61.3635139
2: -24.2974358, 36.9079285, -23.8822231, 36.8920021, -61.1894379, 60.7901535
3: -28.5846634, 41.2655411, -28.1559258, 41.1898422, -69.7745056, 69.4214630
4: -31.5207596, 41.1416130, -31.0554790, 41.0916901, -72.6124496, 72.1970901
5: -27.9455566, 42.5577469, -27.5312767, 42.4836693, -70.4292297, 70.0890198
6: -55.1902924, 27.1306877, -55.0652733, 26.9453621, -82.1356506, 82.1959610
7: -32.3579979, 40.3066101, -31.9968071, 40.3460617, -72.7040558, 72.3034210
8: -37.0558548, 49.2503052, -36.6301079, 49.2066193, -86.2624741, 85.8804169
9: -30.2595081, 38.2328720, -30.1271172, 37.8974533, -68.1569595, 68.3599854
10: -49.4983215, 47.9691162, -49.2973442, 47.2076263, -96.7059479, 97.2664642
11: -48.4572563, 28.9321957, -48.2980042, 28.4340591, -76.8913116, 77.2302017
12: -59.5017204, 31.2048264, -59.5122147, 30.3772354, -89.4564209, 90.2978516
13: -51.1668205, 46.8994980, -51.1172333, 46.6609688, -97.8277893, 98.0167313
14: -79.3050766, 42.5360680, -79.2040329, 41.8872681, -121.1923447, 121.7400970
15: -37.9841118, 35.1056900, -37.6236877, 34.9995193, -72.9836273, 72.7293777
16: -48.5506821, 36.9538651, -48.3618011, 36.6689186, -85.2196045, 85.3156662
17: -79.2204742, 34.0072212, -79.2648010, 33.5067787, -112.7272491, 113.2720184
18: -48.1580353, 33.3552399, -48.0194016, 33.1024780, -81.2605133, 81.3746414
19: -38.2348862, 19.2433510, -38.1129684, 19.1096859, -57.3445740, 57.3563194
20: -34.6687546, 24.9383049, -34.5648384, 24.7600670, -59.4288216, 59.5031433
21: -46.1477509, 24.8450775, -46.0073013, 24.5763817, -70.7241364, 70.8523788
22: -48.8995590, 25.1218758, -48.8308907, 24.9275589, -73.8271179, 73.9527664
23: -37.7925873, 26.3156090, -37.7004395, 26.1868553, -63.9794426, 64.0160522
24: -45.3809967, 28.7927036, -45.1827240, 28.7538948, -74.1348877, 73.9754257
25: -39.5958214, 29.4111748, -39.5338135, 29.2378502, -68.8336716, 68.9449921
26: -55.7642403, 38.6676559, -55.7328568, 38.1531906, -93.9174347, 94.4005127
27: -45.9524155, 30.0195980, -45.6582222, 30.0127392, -75.9651566, 75.6778183
28: -36.9365845, 29.8253632, -36.8195724, 29.7379799, -66.6745605, 66.6449356
29: -51.0010071, 24.5241566, -50.9833984, 24.2549400, -75.2559509, 75.5075531
30: -46.2854652, 33.3206329, -46.1681824, 33.0869904, -79.3724518, 79.4888153
31: -49.0738564, 27.7291870, -48.8926506, 27.5785637, -76.6524200, 76.6218414
32: -55.4961319, 24.5513802, -55.4229774, 24.2779484, -79.6522522, 79.8494263
33: -73.6709518, 31.7064629, -73.3062210, 31.6062527, -104.6977463, 104.3613281
34: -63.6590195, 17.8068542, -63.4575348, 17.7067318, -80.9649124, 80.7646027
35: -60.7395973, 24.3038673, -60.4675522, 24.2136173, -84.3358002, 84.0887909
36: -60.6986732, 25.2587891, -60.6285744, 25.1758919, -85.8732681, 85.8857727
37: -89.3639374, 18.5585041, -89.1892548, 18.3959217, -107.5810471, 107.5702515
38: -69.5729294, 29.0164948, -69.4670486, 28.9235229, -98.4964523, 98.4835434
39: -83.3474503, 30.6652451, -83.1438065, 30.6267262, -113.9741745, 113.8090515
40: -65.7612686, 21.3217659, -65.5463562, 21.3435917, -87.0714722, 86.8244934
41: -58.7144661, 28.5687962, -58.5851593, 28.4875317, -87.2019958, 87.1539536
42: -40.1436119, 24.5923157, -40.0446548, 24.2647667, -64.4083786, 64.6369705

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0723779, upper bound: 37.9691714
time: 84.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 37.9741269
time: 87.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -57.1265297, 42.7889938, -57.0843124, 42.8432770, -99.9698029, 99.8733063
1: -26.4561634, 35.1368332, -26.4289684, 35.2016525, -61.6578140, 61.5658035
2: -24.2974358, 36.9079285, -24.2081776, 36.9669037, -61.2643394, 61.1161041
3: -28.5846634, 41.2655411, -28.5039692, 41.3123970, -69.8970642, 69.7695084
4: -31.5207596, 41.1416130, -31.4117794, 41.1920242, -72.7127838, 72.5533905
5: -27.9455566, 42.5577469, -27.8743267, 42.5993309, -70.5448914, 70.4320755
6: -55.1902924, 27.1306877, -55.1667862, 27.0845947, -82.2748871, 82.2974701
7: -32.3579979, 40.3066101, -32.3125496, 40.4231949, -72.7811890, 72.6191559
8: -37.0558548, 49.2503052, -36.9525909, 49.3083992, -86.3642578, 86.2028961
9: -30.2595081, 38.2328720, -30.2374859, 38.1425896, -68.4020996, 68.4703598
10: -49.4983215, 47.9691162, -49.4965057, 47.7456245, -97.2439423, 97.4656219
11: -48.4572563, 28.9321957, -48.4659386, 28.7967777, -77.2540359, 77.3981323
12: -59.5017204, 31.2048264, -59.6899109, 30.9808331, -90.0662079, 90.4791260
13: -51.1668205, 46.8994980, -51.2300262, 46.8146896, -97.9815063, 98.1295242
14: -79.3050766, 42.5360680, -79.4299164, 42.3191338, -121.6242065, 121.9659882
15: -37.9841118, 35.1056900, -37.8923531, 35.0942879, -73.0783997, 72.9980469
16: -48.5506821, 36.9538651, -48.5302811, 36.9375763, -85.4882584, 85.4841461
17: -79.2204742, 34.0072212, -79.4193726, 33.8503342, -113.0708084, 113.4265900
18: -48.1580353, 33.3552399, -48.1546783, 33.3210640, -81.4790955, 81.5099182
19: -38.2348862, 19.2433510, -38.2229309, 19.2328682, -57.4677544, 57.4662819
20: -34.6687546, 24.9383049, -34.6863365, 24.8968735, -59.5656281, 59.6246414
21: -46.1477509, 24.8450775, -46.1469803, 24.8033695, -70.9511185, 70.9920578
22: -48.8995590, 25.1218758, -48.9492340, 25.0941372, -73.9936981, 74.0711060
23: -37.7925873, 26.3156090, -37.7993011, 26.3005829, -64.0931702, 64.1149139
24: -45.3809967, 28.7927036, -45.3025131, 28.8068104, -74.1878052, 74.0952148
25: -39.5958214, 29.4111748, -39.6182518, 29.3946247, -68.9904480, 69.0294266
26: -55.7642403, 38.6676559, -55.8953400, 38.5806656, -94.3449097, 94.5629959
27: -45.9524155, 30.0195980, -45.8407516, 30.0541630, -76.0065765, 75.8603516
28: -36.9365845, 29.8253632, -36.9095612, 29.8074684, -66.7440491, 66.7349243
29: -51.0010071, 24.5241566, -51.0764465, 24.4807949, -75.4818039, 75.6006012
30: -46.2854652, 33.3206329, -46.2694855, 33.3028336, -79.5883026, 79.5901184
31: -49.0738564, 27.7291870, -49.0431480, 27.7190952, -76.7929535, 76.7723389
32: -55.4961319, 24.5513802, -55.5234604, 24.4717293, -79.8464050, 79.9536133
33: -73.6709518, 31.7064629, -73.5576782, 31.7477322, -104.8322144, 104.6292801
34: -63.6590195, 17.8068542, -63.6103020, 17.8190670, -81.0743790, 80.9447937
35: -60.7395973, 24.3038673, -60.6589203, 24.3205662, -84.4405212, 84.3045349
36: -60.6986732, 25.2587891, -60.7352524, 25.2528992, -85.9503784, 85.9927368
37: -89.3639374, 18.5585041, -89.3197937, 18.5473213, -107.7399368, 107.7055283
38: -69.5729294, 29.0164948, -69.6316757, 29.0098839, -98.5828094, 98.6481705
39: -83.3474503, 30.6652451, -83.2889404, 30.7133045, -114.0607529, 113.9541855
40: -65.7612686, 21.3217659, -65.6979828, 21.4109459, -87.1414185, 86.9808121
41: -58.7144661, 28.5687962, -58.6861000, 28.5909443, -87.3054123, 87.2548981
42: -40.1436119, 24.5923157, -40.1541977, 24.5064316, -64.6500397, 64.7465134

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0105222
time: 84.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0125246
time: 91.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -56.8096161, 42.7044067, -56.9449692, 42.8406906, -99.6503067, 99.6493759
1: -26.2181530, 35.0737915, -26.3161049, 35.2208595, -61.4390106, 61.3898964
2: -23.9247036, 36.8295250, -24.0231209, 37.0173607, -60.9420624, 60.8526459
3: -28.1652756, 41.1394196, -28.2465591, 41.3366394, -69.5019150, 69.3859787
4: -31.1147575, 41.0379143, -31.2236233, 41.2797127, -72.3944702, 72.2615356
5: -27.5322418, 42.4409065, -27.6268044, 42.6229515, -70.1551971, 70.0677109
6: -55.0831375, 26.9682369, -55.2318726, 27.0375061, -82.1206436, 82.2001114
7: -31.9867325, 40.2275925, -32.1369781, 40.4249268, -72.4116592, 72.3645706
8: -36.6964951, 49.1467590, -36.8054771, 49.3929901, -86.0894852, 85.9522400
9: -30.1373901, 37.9848137, -30.3194885, 38.1128769, -68.2502670, 68.3043060
10: -49.2925797, 47.4203262, -49.7750320, 47.6249237, -96.9175034, 97.1953583
11: -48.2840233, 28.5434933, -48.6875191, 28.6763039, -76.9603271, 77.2310104
12: -59.3186569, 30.5392303, -59.9010315, 30.6982918, -89.5915527, 90.0176544
13: -51.0326920, 46.7546387, -51.2187195, 46.8786201, -97.9113159, 97.9733582
14: -79.0570984, 42.0894203, -79.5594635, 42.2046165, -121.2617188, 121.6488800
15: -37.6940956, 35.0046616, -37.8545837, 35.2317581, -72.9258575, 72.8592453
16: -48.3700027, 36.6930580, -48.6812325, 36.9294395, -85.2994385, 85.3742905
17: -79.0492477, 33.6282349, -79.5434723, 33.7455750, -112.7948227, 113.1717072
18: -48.0165215, 33.0823212, -48.1912155, 33.1518211, -81.1683426, 81.2735367
19: -38.1194153, 19.0898952, -38.2322693, 19.1342087, -57.2536240, 57.3221664
20: -34.5386848, 24.7824402, -34.7057648, 24.8234901, -59.3621750, 59.4882050
21: -46.0026321, 24.5771942, -46.2186012, 24.6458740, -70.6485062, 70.7957916
22: -48.7815628, 24.9227104, -48.9898033, 25.0446243, -73.8261871, 73.9125137
23: -37.6892052, 26.1772709, -37.8153305, 26.2257080, -63.9149132, 63.9925995
24: -45.2777023, 28.7283478, -45.3766708, 28.8359566, -74.1136627, 74.1050186
25: -39.5099487, 29.2272911, -39.6366196, 29.3227673, -68.8327179, 68.8639069
26: -55.5930634, 38.1402206, -55.8841209, 38.2378807, -93.8309479, 94.0243378
27: -45.7942772, 29.9682922, -45.9274673, 30.1342506, -75.9285278, 75.8957596
28: -36.8448792, 29.7385178, -36.9049072, 29.8169117, -66.6617889, 66.6434250
29: -50.9061623, 24.2485371, -51.0977440, 24.3213253, -75.2274857, 75.3462830
30: -46.1814804, 33.0642929, -46.2969017, 33.1965294, -79.3780060, 79.3611908
31: -48.9169083, 27.5637646, -49.0732307, 27.6363411, -76.5532532, 76.6369934
32: -55.3928146, 24.3400784, -55.5951385, 24.3996468, -79.6666870, 79.8139191
33: -73.4150848, 31.5525799, -73.5244904, 31.8601990, -104.6980133, 104.4369049
34: -63.5033913, 17.6854038, -63.5969772, 17.9023991, -81.0090942, 80.7915344
35: -60.5569229, 24.1932106, -60.6705322, 24.4752846, -84.4094849, 84.1827545
36: -60.5934448, 25.1595039, -60.7644005, 25.2545204, -85.8461914, 85.9218063
37: -89.2350464, 18.3614140, -89.3462982, 18.4429951, -107.5031891, 107.5250092
38: -69.3955841, 28.9076824, -69.6266174, 28.9969501, -98.3925323, 98.5343018
39: -83.1990662, 30.5697727, -83.3229980, 30.7762489, -113.9753113, 113.8927689
40: -65.6118774, 21.2391739, -65.7491150, 21.3965416, -86.9748383, 86.9447327
41: -58.6078720, 28.4376659, -58.7042885, 28.5178947, -87.1257629, 87.1419525
42: -40.0312271, 24.3022213, -40.2757797, 24.3964920, -64.4277191, 64.5780029

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 725
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
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 679
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 608
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 537
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0034228
time: 91.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0109895
time: 90.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -56.8842697, 42.7207413, -57.1657143, 42.9176407, -99.8019104, 99.8864594
1: -26.2711945, 35.0849609, -26.4712048, 35.2769699, -61.5481644, 61.5561676
2: -24.0142460, 36.8408813, -24.2844791, 37.1000900, -61.1143341, 61.1253586
3: -28.2648354, 41.1586685, -28.5369835, 41.4475021, -69.7123413, 69.6956482
4: -31.2064705, 41.0529861, -31.4923477, 41.3744888, -72.5809631, 72.5453339
5: -27.6286659, 42.4584503, -27.9099159, 42.7353821, -70.3640442, 70.3683624
6: -55.1084747, 27.0116234, -55.3135033, 27.1630592, -82.2715302, 82.3251266
7: -32.0702591, 40.2409859, -32.3846626, 40.4990082, -72.5692673, 72.6256485
8: -36.7803574, 49.1627197, -37.0528221, 49.4844322, -86.2647858, 86.2155457
9: -30.1641960, 38.0379333, -30.4012394, 38.2753487, -68.4395447, 68.4391708
10: -49.3228149, 47.5355835, -49.9133911, 47.9663963, -97.2892151, 97.4489746
11: -48.3106918, 28.6331348, -48.8255005, 28.9250050, -77.2356949, 77.4586334
12: -59.3400192, 30.7056885, -60.0667000, 31.1848831, -90.1018066, 90.3616562
13: -51.0651550, 46.7930946, -51.3075027, 47.0030556, -98.0682068, 98.1006012
14: -79.1013412, 42.1986313, -79.7440109, 42.5242615, -121.6256027, 121.9426422
15: -37.7611961, 35.0297699, -38.0394821, 35.3019753, -73.0631714, 73.0692520
16: -48.4085617, 36.7446747, -48.8092957, 37.0759430, -85.4845047, 85.5539703
17: -79.0796890, 33.7303467, -79.7043991, 34.0427589, -113.1224518, 113.4347458
18: -48.0458069, 33.1485443, -48.3086662, 33.3477631, -81.3935699, 81.4572144
19: -38.1425552, 19.1260471, -38.3344727, 19.2401047, -57.3826599, 57.4605179
20: -34.5641632, 24.8217354, -34.8057022, 24.9377747, -59.5019379, 59.6274376
21: -46.0275955, 24.6387253, -46.3412857, 24.8265705, -70.8541641, 70.9800110
22: -48.8074379, 24.9730511, -49.0744247, 25.1965866, -74.0040283, 74.0474777
23: -37.7086716, 26.2061710, -37.8994446, 26.3109322, -64.0196075, 64.1056137
24: -45.3064041, 28.7456589, -45.4658699, 28.8858109, -74.1922150, 74.2115326
25: -39.5312386, 29.2667274, -39.7045593, 29.4409733, -68.9722137, 68.9712830
26: -55.6230545, 38.2699547, -56.0354691, 38.6140594, -94.2371140, 94.3054199
27: -45.8302879, 29.9841118, -46.0453796, 30.1761818, -76.0064697, 76.0294952
28: -36.8685684, 29.7644215, -36.9971123, 29.8910427, -66.7596130, 66.7615356
29: -50.9289131, 24.3181934, -51.1888390, 24.5201015, -75.4490128, 75.5070343
30: -46.2026405, 33.1188889, -46.3716888, 33.3545418, -79.5571823, 79.4905777
31: -48.9496460, 27.5961037, -49.1905022, 27.7285423, -76.6781921, 76.7866058
32: -55.4146385, 24.3981323, -55.6868591, 24.5699558, -79.8590393, 79.9701843
33: -73.4784775, 31.5811138, -73.7127686, 31.9767647, -104.8881531, 104.6521912
34: -63.5352821, 17.7076263, -63.6952477, 17.9803104, -81.1345749, 80.9195404
35: -60.5926743, 24.2101822, -60.7757111, 24.5410404, -84.5250549, 84.3149719
36: -60.6125336, 25.1922970, -60.8262215, 25.3509369, -85.9625397, 86.0173035
37: -89.2679825, 18.4210701, -89.4752274, 18.6231632, -107.7194672, 107.7249603
38: -69.4301605, 28.9438457, -69.7412262, 29.1109772, -98.5411377, 98.6850739
39: -83.2318573, 30.5877495, -83.4247131, 30.8452377, -114.0770950, 114.0124664
40: -65.6445770, 21.2586517, -65.8551178, 21.4566574, -87.0718155, 87.0760040
41: -58.6338272, 28.4730530, -58.7879982, 28.6178837, -87.2517090, 87.2610474
42: -40.0508919, 24.3652897, -40.3695755, 24.5880909, -64.6389847, 64.7348633

Time for backsubstitution: 2.22 seconds

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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 725
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
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 679
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 705
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
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 971
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
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 608
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 606
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
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1633

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1116272
time: 96.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1191533
time: 92.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -57.1560478, 42.8019333, -56.9829521, 42.8558044, -100.0118561, 99.7848816
1: -26.4744949, 35.1458282, -26.3347702, 35.2283707, -61.7028656, 61.4805984
2: -24.3409119, 36.9158707, -24.0656376, 37.0305939, -61.3715057, 60.9815063
3: -28.6189480, 41.2813873, -28.3007507, 41.3403816, -69.9593277, 69.5821381
4: -31.5634422, 41.1540947, -31.2522812, 41.2916145, -72.8550568, 72.4063721
5: -27.9775429, 42.5737381, -27.6764431, 42.6349258, -70.6124725, 70.2501831
6: -55.2106209, 27.1555290, -55.2420998, 27.0661850, -82.2768097, 82.3976288
7: -32.3881912, 40.3182755, -32.1709518, 40.4341354, -72.8223267, 72.4892273
8: -37.1039581, 49.2645721, -36.8350143, 49.4031258, -86.5070801, 86.0995865
9: -30.2744064, 38.2856064, -30.3210526, 38.1100540, -68.3844604, 68.6066589
10: -49.5214691, 48.0761070, -49.7541275, 47.6026917, -97.1241608, 97.8302307
11: -48.4781914, 29.0017586, -48.6905670, 28.6846466, -77.1628418, 77.6923218
12: -59.5184212, 31.3106728, -59.9130783, 30.7785416, -89.8732147, 90.8090363
13: -51.1794434, 46.9485588, -51.2352142, 46.8935280, -98.0729675, 98.1837769
14: -79.3284149, 42.6317863, -79.5692291, 42.2318611, -121.5602722, 122.2010193
15: -38.0432243, 35.1242218, -37.8631210, 35.2294769, -73.2727051, 72.9873428
16: -48.5774117, 37.0162773, -48.6882172, 36.9053268, -85.4827423, 85.7044983
17: -79.2351913, 34.0827446, -79.5709839, 33.8049240, -113.0401154, 113.6537323
18: -48.1817970, 33.3723907, -48.2067223, 33.1983032, -81.3800964, 81.5791168
19: -38.2522888, 19.2502956, -38.2517548, 19.1582451, -57.4105339, 57.5020523
20: -34.6858063, 24.9594536, -34.7149086, 24.8455086, -59.5313148, 59.6743622
21: -46.1672249, 24.8661461, -46.2298241, 24.6740456, -70.8412704, 71.0959702
22: -48.9276733, 25.1414948, -48.9876213, 25.0838890, -74.0115662, 74.1291199
23: -37.8077011, 26.3222179, -37.8231812, 26.2334709, -64.0411682, 64.1454010
24: -45.4281311, 28.7999344, -45.3815041, 28.8504829, -74.2786102, 74.1814423
25: -39.6156387, 29.4245300, -39.6444817, 29.3342857, -68.9499207, 69.0690155
26: -55.7863121, 38.7066345, -55.9028091, 38.3226051, -94.1089172, 94.6094437
27: -46.0144768, 30.0263462, -45.9191856, 30.1538010, -76.1682739, 75.9455338
28: -36.9593086, 29.8344421, -36.9333344, 29.8423805, -66.8016891, 66.7677765
29: -51.0225983, 24.5470028, -51.1229172, 24.3679276, -75.3905258, 75.6699219
30: -46.3037415, 33.3491135, -46.2943459, 33.2044067, -79.5081482, 79.6434631
31: -49.0996742, 27.7390900, -49.0794182, 27.6359062, -76.7355804, 76.8185120
32: -55.5151176, 24.5934963, -55.6137657, 24.4410439, -79.8338852, 80.0861969
33: -73.7312164, 31.7223740, -73.5409393, 31.8685112, -105.0552673, 104.6211472
34: -63.6891670, 17.8196468, -63.5905151, 17.8970413, -81.2371979, 80.9250946
35: -60.7853699, 24.3161297, -60.6481323, 24.4563484, -84.6605759, 84.2892914
36: -60.7234612, 25.2691460, -60.7483101, 25.2991295, -86.0217819, 86.0159912
37: -89.4003906, 18.5728912, -89.3828964, 18.5182018, -107.7465820, 107.7814255
38: -69.5979919, 29.0303211, -69.6249313, 29.0491085, -98.6471024, 98.6552505
39: -83.3783569, 30.6743183, -83.3219452, 30.7798061, -114.1581650, 113.9962616
40: -65.7979584, 21.3264389, -65.7472000, 21.4120083, -87.1835327, 87.0351334
41: -58.7355118, 28.5777321, -58.7194176, 28.5500698, -87.2855835, 87.2971497
42: -40.1606522, 24.6290760, -40.2815094, 24.4130859, -64.5737381, 64.9105835

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0059000
time: 97.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1851302, upper bound: 38.0110601
time: 94.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -57.1560478, 42.8019333, -57.2557449, 42.9386978, -100.0947418, 100.0576782
1: -26.4744949, 35.1458282, -26.5368557, 35.2912140, -61.7657089, 61.6826859
2: -24.3409119, 36.9158707, -24.3935165, 37.1136208, -61.4545326, 61.3093872
3: -28.6189480, 41.2813873, -28.6563625, 41.4715500, -70.0904999, 69.9377518
4: -31.5634422, 41.1540947, -31.6079006, 41.3935547, -72.9570007, 72.7619934
5: -27.9775429, 42.5737381, -28.0257626, 42.7566147, -70.7341614, 70.5995026
6: -55.2106209, 27.1555290, -55.3475571, 27.2117577, -82.4223785, 82.5030823
7: -32.3881912, 40.3182755, -32.4874344, 40.5157890, -72.9039764, 72.8057098
8: -37.1039581, 49.2645721, -37.1574097, 49.5053482, -86.6093063, 86.4219818
9: -30.2744064, 38.2856064, -30.4324799, 38.3562851, -68.6306915, 68.7180862
10: -49.5214691, 48.0761070, -49.9533310, 48.1398087, -97.6612778, 98.0294342
11: -48.4781914, 29.0017586, -48.8591499, 29.0494976, -77.5276871, 77.8609085
12: -59.5184212, 31.3106728, -60.0930939, 31.3818264, -90.4827118, 90.9925232
13: -51.1794434, 46.9485588, -51.3475456, 47.0517120, -98.2311554, 98.2961044
14: -79.3284149, 42.6317863, -79.7976990, 42.6646156, -121.9930267, 122.4294891
15: -38.0432243, 35.1242218, -38.1415405, 35.3344955, -73.3777161, 73.2657623
16: -48.5774117, 37.0162773, -48.8572121, 37.1747742, -85.7521820, 85.8734894
17: -79.2351913, 34.0827446, -79.7371140, 34.1592827, -113.3944702, 113.8198547
18: -48.1817970, 33.3723907, -48.3456383, 33.4235573, -81.6053543, 81.7180328
19: -38.2522888, 19.2502956, -38.3633385, 19.2814293, -57.5337181, 57.6136322
20: -34.6858063, 24.9594536, -34.8375206, 24.9829292, -59.6687355, 59.7969742
21: -46.1672249, 24.8661461, -46.3726425, 24.9018059, -71.0690308, 71.2387848
22: -48.9276733, 25.1414948, -49.1087189, 25.2514210, -74.1790924, 74.2502136
23: -37.8077011, 26.3222179, -37.9235916, 26.3499737, -64.1576767, 64.2458115
24: -45.4281311, 28.7999344, -45.5044174, 28.9038506, -74.3319855, 74.3043518
25: -39.6156387, 29.4245300, -39.7305756, 29.4914284, -69.1070709, 69.1551056
26: -55.7863121, 38.7066345, -56.0725975, 38.7609291, -94.5472412, 94.7792358
27: -46.0144768, 30.0263462, -46.1036682, 30.1956463, -76.2101212, 76.1300125
28: -36.9593086, 29.8344421, -37.0252457, 29.9120445, -66.8713531, 66.8596878
29: -51.0225983, 24.5470028, -51.2179222, 24.5961800, -75.6187744, 75.7649231
30: -46.3037415, 33.3491135, -46.3993950, 33.4366913, -79.7404327, 79.7485046
31: -49.0996742, 27.7390900, -49.2319565, 27.7776833, -76.8773575, 76.9710464
32: -55.5151176, 24.5934963, -55.7152863, 24.6376057, -80.0305328, 80.1914215
33: -73.7312164, 31.7223740, -73.7940979, 32.0098267, -105.1894150, 104.8907623
34: -63.6891670, 17.8196468, -63.7445526, 18.0089417, -81.3459320, 81.1064987
35: -60.7853699, 24.3161297, -60.8406372, 24.5632000, -84.7647476, 84.5056381
36: -60.7234612, 25.2691460, -60.8592606, 25.3759804, -86.0987396, 86.1272736
37: -89.4003906, 18.5728912, -89.5172043, 18.6660786, -107.9006805, 107.9208908
38: -69.5979919, 29.0303211, -69.7917328, 29.1358566, -98.7338486, 98.8220520
39: -83.3783569, 30.6743183, -83.4691086, 30.8663139, -114.2446747, 114.1434250
40: -65.7979584, 21.3264389, -65.9013062, 21.4792652, -87.2534485, 87.1939850
41: -58.7355118, 28.5777321, -58.8215294, 28.6541214, -87.3896332, 87.3992615
42: -40.1606522, 24.6290760, -40.3948860, 24.6748638, -64.8355179, 65.0239639

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0499484
time: 205.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0516191
time: 90.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -57.0042648, 42.8590965, -56.7885246, 42.7798233, -99.7840881, 99.6476212
1: -26.3717213, 35.2216492, -26.2151871, 35.1646767, -61.5363998, 61.4368362
2: -24.0720482, 36.9627495, -23.8458061, 36.9012222, -60.9732704, 60.8085556
3: -28.3281574, 41.2974854, -28.1031666, 41.2118988, -69.5400543, 69.4006500
4: -31.2470741, 41.1770096, -31.0306625, 41.1096115, -72.3566895, 72.2076721
5: -27.6721363, 42.5851517, -27.4837761, 42.4963799, -70.1685181, 70.0689240
6: -55.1414871, 27.0480366, -55.0604706, 26.9242783, -82.0657654, 82.1085052
7: -32.2461700, 40.4644394, -31.9716263, 40.3891525, -72.6353226, 72.4360657
8: -36.8159828, 49.3141556, -36.6032448, 49.2319183, -86.0478973, 85.9174042
9: -30.2319412, 38.0491791, -30.1272850, 37.9095383, -68.1414795, 68.1764679
10: -49.4080887, 47.5506287, -49.3394623, 47.2431107, -96.6511993, 96.8900909
11: -48.3652611, 28.5465832, -48.3129425, 28.4151497, -76.7804108, 76.8595276
12: -59.6899910, 30.9406471, -59.5875511, 30.3084717, -89.5658188, 90.1135406
13: -51.2197685, 46.9629974, -51.1365700, 46.6560135, -97.8757782, 98.0995636
14: -79.4058914, 42.3673668, -79.2633057, 41.8648491, -121.2707367, 121.6306763
15: -37.7705269, 35.1100464, -37.6091843, 35.0005226, -72.7710495, 72.7192307
16: -48.6438599, 36.8839531, -48.3688049, 36.7370491, -85.3809052, 85.2527618
17: -79.4602127, 33.9473495, -79.3188095, 33.4454956, -112.9057083, 113.2661591
18: -48.1850281, 33.1466141, -48.0280914, 33.0554428, -81.2404709, 81.1747055
19: -38.1855621, 19.0864658, -38.1058807, 19.0712624, -57.2568245, 57.1923447
20: -34.6672440, 24.8620300, -34.5756035, 24.7409096, -59.4081535, 59.4376335
21: -46.0954704, 24.5949974, -46.0094757, 24.5354633, -70.6309357, 70.6044769
22: -49.0304298, 25.1431389, -48.8840828, 24.8933697, -73.9237976, 74.0272217
23: -37.7866364, 26.1818905, -37.7077637, 26.1705894, -63.9572258, 63.8896561
24: -45.3897896, 28.7927303, -45.1892090, 28.7482758, -74.1380615, 73.9819412
25: -39.6551094, 29.3521290, -39.5557175, 29.2315922, -68.8867035, 68.9078445
26: -55.9377174, 38.4675484, -55.7804832, 38.0681686, -94.0058899, 94.2480316
27: -45.9093475, 30.0562420, -45.6781998, 30.0123634, -75.9217072, 75.7344437
28: -36.9239044, 29.7586021, -36.8023758, 29.7051926, -66.6290970, 66.5609741
29: -51.1440430, 24.4771767, -51.0102654, 24.2115421, -75.3555832, 75.4874420
30: -46.2804413, 33.1650581, -46.1758575, 33.0852127, -79.3656540, 79.3409119
31: -49.0213547, 27.6006489, -48.8998489, 27.5733528, -76.5947113, 76.5004959
32: -55.5226364, 24.4644566, -55.4297752, 24.2421227, -79.6373901, 79.7737885
33: -73.5657730, 31.7041950, -73.2994537, 31.6261482, -104.6608124, 104.3508606
34: -63.5804176, 17.7811413, -63.4710121, 17.7287273, -80.9956284, 80.7504654
35: -60.6265602, 24.2668438, -60.5002670, 24.2468948, -84.2907410, 84.0783081
36: -60.7723351, 25.3545399, -60.6813622, 25.1364441, -85.9065933, 86.0342255
37: -89.3756638, 18.4134502, -89.1710052, 18.3249550, -107.5181198, 107.4031601
38: -69.6358109, 29.1034241, -69.5158997, 28.8804817, -98.5162964, 98.6193237
39: -83.2910004, 30.6860275, -83.1540375, 30.6461887, -113.9371872, 113.8400650
40: -65.7410965, 21.4058189, -65.5532074, 21.3661575, -87.0878296, 86.9077148
41: -58.6981010, 28.5327377, -58.5763512, 28.4708271, -87.1689301, 87.1090851
42: -40.1250763, 24.4199772, -40.0533218, 24.2473030, -64.3723755, 64.4732971

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
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
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1699
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 606
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 37.9926060
time: 87.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0004540
time: 93.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -57.0790634, 42.8754807, -57.0093498, 42.8567200, -99.9357834, 99.8848267
1: -26.4247913, 35.2328110, -26.3701954, 35.2212791, -61.6460724, 61.6030045
2: -24.1618156, 36.9740524, -24.1070156, 36.9842453, -61.1460609, 61.0810699
3: -28.4279308, 41.3166466, -28.3936520, 41.3229637, -69.7508926, 69.7102966
4: -31.3390617, 41.1922760, -31.2990799, 41.2047844, -72.5438461, 72.4913559
5: -27.7688713, 42.6025734, -27.7665138, 42.6092758, -70.3781433, 70.3690872
6: -55.1670341, 27.0906391, -55.1402283, 27.0435257, -82.2105560, 82.2308655
7: -32.3307495, 40.4778023, -32.2195511, 40.4635315, -72.7942810, 72.6973572
8: -36.9001389, 49.3301163, -36.8505058, 49.3239441, -86.2240829, 86.1806183
9: -30.2590981, 38.1022491, -30.2099228, 38.0716171, -68.3307190, 68.3121719
10: -49.4380798, 47.6664734, -49.4781647, 47.5844727, -97.0225525, 97.1446381
11: -48.3921471, 28.6355228, -48.4515991, 28.6635208, -77.0556641, 77.0871201
12: -59.7112503, 31.1075363, -59.7540970, 30.7948151, -90.0757828, 90.4589844
13: -51.2521744, 47.0021057, -51.2252388, 46.7794609, -98.0316315, 98.2273407
14: -79.4500961, 42.4770088, -79.4484100, 42.1842957, -121.6343918, 121.9254150
15: -37.8355980, 35.1356888, -37.7899017, 35.0720520, -72.9076538, 72.9255905
16: -48.6827431, 36.9346581, -48.4974365, 36.8818741, -85.5646210, 85.4320984
17: -79.4906464, 34.0507088, -79.4806213, 33.7446747, -113.2353210, 113.5313263
18: -48.2145996, 33.2129898, -48.1436653, 33.2511673, -81.4657669, 81.3566589
19: -38.2088699, 19.1227264, -38.2078018, 19.1773186, -57.3861885, 57.3305283
20: -34.6927490, 24.9014282, -34.6761169, 24.8549690, -59.5477180, 59.5775452
21: -46.1207161, 24.6566162, -46.1329727, 24.7162590, -70.8369751, 70.7895889
22: -49.0561447, 25.1945381, -48.9680977, 25.0455990, -74.1017456, 74.1626358
23: -37.8062134, 26.2108307, -37.7920151, 26.2547226, -64.0609360, 64.0028458
24: -45.4188614, 28.8100815, -45.2773323, 28.7982140, -74.2170715, 74.0874176
25: -39.6761703, 29.3922806, -39.6221390, 29.3498192, -69.0259857, 69.0144196
26: -55.9676514, 38.5979385, -55.9321098, 38.4459534, -94.4136047, 94.5300446
27: -45.9459839, 30.0721569, -45.7953873, 30.0543213, -76.0003052, 75.8675461
28: -36.9477043, 29.7847290, -36.8937073, 29.7818451, -66.7295532, 66.6784363
29: -51.1669083, 24.5471382, -51.1008530, 24.4102936, -75.5772018, 75.6479950
30: -46.3017883, 33.2175217, -46.2506638, 33.2366600, -79.5384521, 79.4681854
31: -49.0548668, 27.6324368, -49.0165634, 27.6643562, -76.7192230, 76.6490021
32: -55.5444641, 24.5226612, -55.5220299, 24.4119492, -79.8292694, 79.9307404
33: -73.6293869, 31.7327232, -73.4875488, 31.7433376, -104.8518677, 104.5659332
34: -63.6124878, 17.8033943, -63.5690765, 17.8070202, -81.1221466, 80.8780975
35: -60.6624870, 24.2838154, -60.6050911, 24.3131523, -84.4071121, 84.2101593
36: -60.7912827, 25.3875389, -60.7424316, 25.2343483, -86.0240784, 86.1291046
37: -89.4085388, 18.4732189, -89.2978973, 18.5050049, -107.7346039, 107.6011505
38: -69.6701813, 29.1397076, -69.6302490, 28.9945927, -98.6647720, 98.7699585
39: -83.3239746, 30.7040443, -83.2552795, 30.7153473, -114.0393219, 113.9593201
40: -65.7742538, 21.4251041, -65.6586914, 21.4260845, -87.1849518, 87.0382843
41: -58.7244263, 28.5679779, -58.6603279, 28.5687447, -87.2931671, 87.2283020
42: -40.1449013, 24.4839211, -40.1478043, 24.4372635, -64.5821686, 64.6317291

Time for backsubstitution: 2.31 seconds

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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1686
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1065152
time: 88.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1147878
time: 92.22 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -57.3484917, 42.9565430, -56.8297424, 42.7964249, -100.1449127, 99.7862854
1: -26.6275597, 35.2934380, -26.2335911, 35.1743851, -61.8019447, 61.5270309
2: -24.4891663, 37.0486946, -23.8875904, 36.9228134, -61.4119797, 60.9362869
3: -28.7794418, 41.4388657, -28.1597042, 41.2241478, -70.0035858, 69.5985718
4: -31.6969833, 41.2926750, -31.0583096, 41.1229401, -72.8199234, 72.3509827
5: -28.1156006, 42.7175446, -27.5368385, 42.5149498, -70.6305542, 70.2543793
6: -55.2686234, 27.2326241, -55.0726624, 26.9499607, -82.2185822, 82.3052826
7: -32.6485748, 40.5545502, -32.0062714, 40.4030075, -73.0515823, 72.5608215
8: -37.2240105, 49.4317780, -36.6323700, 49.2432709, -86.4672852, 86.0641479
9: -30.3696594, 38.3492241, -30.1307812, 37.9070854, -68.2767487, 68.4800034
10: -49.6351814, 48.2078323, -49.3195572, 47.2200813, -96.8552628, 97.5273895
11: -48.5603104, 29.0014858, -48.3170128, 28.4251823, -76.9854889, 77.3184967
12: -59.8882751, 31.7134399, -59.6024628, 30.3880539, -89.8454285, 90.9091797
13: -51.3656998, 47.1582146, -51.1524353, 46.6727448, -98.0384445, 98.3106537
14: -79.6757050, 42.9105453, -79.2755814, 41.8922272, -121.5679321, 122.1861267
15: -38.1107216, 35.2311821, -37.6213188, 35.0098228, -73.1205444, 72.8525009
16: -48.8522377, 37.2063026, -48.3765678, 36.7114716, -85.5637054, 85.5828705
17: -79.6452637, 34.3996658, -79.3579712, 33.5121384, -113.1574020, 113.7576370
18: -48.3509331, 33.4344902, -48.0440331, 33.1057549, -81.4566879, 81.4785233
19: -38.3189926, 19.2466869, -38.1260986, 19.0952759, -57.4142685, 57.3727875
20: -34.8138695, 25.0393028, -34.5862961, 24.7630424, -59.5769119, 59.6255989
21: -46.2603226, 24.8841476, -46.0241661, 24.5632420, -70.8235626, 70.9083099
22: -49.1757812, 25.3630943, -48.8835182, 24.9335823, -74.1093597, 74.2466125
23: -37.9051514, 26.3253422, -37.7168427, 26.1780472, -64.0831985, 64.0421829
24: -45.5410843, 28.8632832, -45.1950722, 28.7632713, -74.3043518, 74.0583572
25: -39.7597961, 29.5504341, -39.5631485, 29.2435913, -69.0033875, 69.1135864
26: -56.1295738, 39.0313721, -55.8060837, 38.1585121, -94.2880859, 94.8374557
27: -46.1314774, 30.1140690, -45.6703300, 30.0322113, -76.1636887, 75.7844009
28: -37.0387268, 29.8547726, -36.8310928, 29.7326965, -66.7714233, 66.6858673
29: -51.2602043, 24.7753887, -51.0362511, 24.2591286, -75.5193329, 75.8116379
30: -46.4031906, 33.4421387, -46.1766853, 33.0965805, -79.4997711, 79.6188202
31: -49.2070427, 27.7738228, -48.9066391, 27.5729122, -76.7799530, 76.6804657
32: -55.6444092, 24.7183838, -55.4498177, 24.2832069, -79.8037262, 80.0480957
33: -73.8820496, 31.8735275, -73.3168564, 31.6350365, -105.0187988, 104.5354538
34: -63.7663689, 17.9151230, -63.4651489, 17.7232132, -81.2244186, 80.8839722
35: -60.8552742, 24.3897648, -60.4782753, 24.2284851, -84.5425491, 84.1848068
36: -60.9009781, 25.4647846, -60.6675453, 25.1822567, -86.0817032, 86.1312408
37: -89.5405045, 18.6255589, -89.2079697, 18.3977757, -107.7602234, 107.6603775
38: -69.8366547, 29.2264977, -69.5156784, 28.9330044, -98.7696609, 98.7421722
39: -83.4706955, 30.7904606, -83.1538086, 30.6499348, -114.1206284, 113.9442673
40: -65.9274139, 21.4932652, -65.5524063, 21.3811531, -87.2964630, 86.9989929
41: -58.8261414, 28.6722794, -58.5927048, 28.5013809, -87.3275223, 87.2649841
42: -40.2539520, 24.7376537, -40.0630875, 24.2730236, -64.5269775, 64.8007431

Time for backsubstitution: 2.24 seconds

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
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0311844
time: 84.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0372138
time: 88.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -57.3484917, 42.9565430, -57.0953827, 42.8774033, -100.2258911, 100.0519257
1: -26.6275597, 35.2934380, -26.4358521, 35.2355003, -61.8630600, 61.7292900
2: -24.4891663, 37.0486946, -24.2135181, 36.9977417, -61.4869080, 61.2622147
3: -28.7794418, 41.4388657, -28.5077457, 41.3467255, -70.1261673, 69.9466095
4: -31.6969833, 41.2926750, -31.4145508, 41.2233810, -72.9203644, 72.7072296
5: -28.1156006, 42.7175446, -27.8798027, 42.6305885, -70.7461853, 70.5973511
6: -55.2686234, 27.2326241, -55.1742897, 27.0897942, -82.3584137, 82.4069138
7: -32.6485748, 40.5545502, -32.3218918, 40.4801178, -73.1286926, 72.8764420
8: -37.2240105, 49.4317780, -36.9548569, 49.3450737, -86.5690842, 86.3866348
9: -30.3696594, 38.3492241, -30.2412586, 38.1521759, -68.5218353, 68.5904846
10: -49.6351814, 48.2078323, -49.5180473, 47.7578888, -97.3930664, 97.7258759
11: -48.5603104, 29.0014858, -48.4851074, 28.7874947, -77.3478088, 77.4865952
12: -59.8882751, 31.7134399, -59.7801514, 30.9915733, -90.4550934, 91.0904388
13: -51.3656998, 47.1582146, -51.2651138, 46.8266640, -98.1923676, 98.4233246
14: -79.6757050, 42.9105453, -79.5013733, 42.3241119, -121.9998169, 122.4119186
15: -38.1107216, 35.2311821, -37.8888283, 35.1047897, -73.2155151, 73.1200104
16: -48.8522377, 37.2063026, -48.5453186, 36.9799576, -85.8321991, 85.7516174
17: -79.6452637, 34.3996658, -79.5125122, 33.8557396, -113.5010071, 113.9121780
18: -48.3509331, 33.4344902, -48.1793022, 33.3244705, -81.6753998, 81.6137924
19: -38.3189926, 19.2466869, -38.2362671, 19.2188015, -57.5377960, 57.4829559
20: -34.8138695, 25.0393028, -34.7077713, 24.8997917, -59.7136612, 59.7470741
21: -46.2603226, 24.8841476, -46.1640549, 24.7900925, -71.0504150, 71.0482025
22: -49.1757812, 25.3630943, -49.0016708, 25.1001320, -74.2759094, 74.3647614
23: -37.9051514, 26.3253422, -37.8158340, 26.2917938, -64.1969452, 64.1411743
24: -45.5410843, 28.8632832, -45.3148308, 28.8161659, -74.3572540, 74.1781158
25: -39.7597961, 29.5504341, -39.6471062, 29.4002686, -69.1600647, 69.1975403
26: -56.1295738, 39.0313721, -55.9684715, 38.5860672, -94.7156372, 94.9998474
27: -46.1314774, 30.1140690, -45.8529396, 30.0736618, -76.2051392, 75.9670105
28: -37.0387268, 29.8547726, -36.9211235, 29.8024559, -66.8411865, 66.7758942
29: -51.2602043, 24.7753887, -51.1293030, 24.4849033, -75.7451096, 75.9046936
30: -46.4031906, 33.4421387, -46.2781029, 33.3122711, -79.7154617, 79.7202454
31: -49.2070427, 27.7738228, -49.0572548, 27.7132492, -76.9202881, 76.8310776
32: -55.6444092, 24.7183838, -55.5503616, 24.4768467, -79.9977570, 80.1523285
33: -73.8820496, 31.8735275, -73.5682526, 31.7764416, -105.1532745, 104.8033752
34: -63.7663689, 17.9151230, -63.6180649, 17.8355942, -81.3338623, 81.0641632
35: -60.8552742, 24.3897648, -60.6696243, 24.3355331, -84.6473236, 84.4005127
36: -60.9009781, 25.4647846, -60.7740097, 25.2594032, -86.1589890, 86.2380066
37: -89.5405045, 18.6255589, -89.3386536, 18.5492992, -107.9193115, 107.7959595
38: -69.8366547, 29.2264977, -69.6799622, 29.0194950, -98.8561478, 98.9064636
39: -83.4706955, 30.7904606, -83.2990570, 30.7363529, -114.2070465, 114.0895157
40: -65.9274139, 21.4932652, -65.7040100, 21.4487152, -87.3665161, 87.1554565
41: -58.8261414, 28.6722794, -58.6936188, 28.6047478, -87.4308929, 87.3658981
42: -40.2539520, 24.7376537, -40.1727829, 24.5141220, -64.7680740, 64.9104385

Time for backsubstitution: 2.22 seconds

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
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
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
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0745986
time: 91.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0769717
time: 88.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -57.0307121, 42.8720322, -56.9559631, 42.8747444, -99.9054565, 99.8279953
1: -26.3897209, 35.2305832, -26.3229485, 35.2546196, -61.6443405, 61.5535316
2: -24.1155071, 36.9705048, -24.0284348, 37.0481682, -61.1636734, 60.9989395
3: -28.3593941, 41.3132553, -28.2503109, 41.3707542, -69.7301483, 69.5635681
4: -31.2896519, 41.1885262, -31.2263737, 41.3109550, -72.6006088, 72.4149017
5: -27.7013836, 42.6012115, -27.6322613, 42.6541824, -70.3555679, 70.2334747
6: -55.1610107, 27.0684338, -55.2392921, 27.0419464, -82.2029572, 82.3077240
7: -32.2759476, 40.4760513, -32.1462708, 40.4816780, -72.7576294, 72.6223221
8: -36.8635712, 49.3284569, -36.8076859, 49.4296227, -86.2931976, 86.1361389
9: -30.2468243, 38.1012497, -30.3232136, 38.1226273, -68.3694534, 68.4244614
10: -49.4309425, 47.6574593, -49.7971115, 47.6373520, -97.0682983, 97.4545746
11: -48.3863411, 28.6153336, -48.7064667, 28.6673927, -77.0537338, 77.3218002
12: -59.7060242, 31.0465889, -59.9910126, 30.7090416, -89.9812012, 90.6274567
13: -51.2321091, 47.0116577, -51.2539024, 46.8900337, -98.1221466, 98.2655640
14: -79.4287949, 42.4629898, -79.6307983, 42.2094574, -121.6382523, 122.0937881
15: -37.8257065, 35.1289291, -37.8537064, 35.2421455, -73.0678558, 72.9826355
16: -48.6703949, 36.9457283, -48.6960983, 36.9718781, -85.6422729, 85.6418304
17: -79.4746094, 34.0195465, -79.6363678, 33.7507362, -113.2253418, 113.6559143
18: -48.2088318, 33.1611710, -48.2153511, 33.1550674, -81.3638992, 81.3765259
19: -38.2030907, 19.0927696, -38.2452850, 19.1199341, -57.3230247, 57.3380547
20: -34.6842346, 24.8830414, -34.7271576, 24.8264656, -59.5107002, 59.6101990
21: -46.1145554, 24.6161976, -46.2354927, 24.6327152, -70.7472687, 70.8516922
22: -49.0578575, 25.1623077, -49.0418472, 25.0506439, -74.1085052, 74.2041550
23: -37.8013496, 26.1868172, -37.8316231, 26.2169132, -64.0182648, 64.0184402
24: -45.4365997, 28.7990017, -45.3887787, 28.8453560, -74.2819519, 74.1877823
25: -39.6745224, 29.3655319, -39.6656494, 29.3284454, -69.0029678, 69.0311813
26: -55.9595490, 38.5034790, -55.9568825, 38.2432098, -94.2027588, 94.4603577
27: -45.9711456, 30.0625858, -45.9393349, 30.1537666, -76.1249084, 76.0019226
28: -36.9464035, 29.7672634, -36.9162750, 29.8117371, -66.7581406, 66.6835403
29: -51.1654663, 24.4988995, -51.1504440, 24.3254013, -75.4908676, 75.6493454
30: -46.2987518, 33.1860962, -46.3054008, 33.2058640, -79.5046158, 79.4915009
31: -49.0480499, 27.6097412, -49.0869522, 27.6316185, -76.6796722, 76.6966934
32: -55.5413208, 24.5062866, -55.6219940, 24.4047928, -79.8183670, 80.0117798
33: -73.6255722, 31.7202854, -73.5350647, 31.8888988, -105.0183563, 104.6117401
34: -63.6101608, 17.7940331, -63.6045303, 17.9188957, -81.2674255, 80.9115753
35: -60.6720505, 24.2793636, -60.6812248, 24.4901371, -84.6156235, 84.2789764
36: -60.7959442, 25.3649311, -60.8031349, 25.2613411, -86.0555038, 86.1666031
37: -89.4113922, 18.4278316, -89.3647232, 18.4449081, -107.6814194, 107.6143036
38: -69.6595764, 29.1172848, -69.6749191, 29.0065155, -98.6660919, 98.7922058
39: -83.3216705, 30.6950989, -83.3328781, 30.7995148, -114.1211853, 114.0279770
40: -65.7769852, 21.4103203, -65.7549896, 21.4342575, -87.1990051, 87.1191559
41: -58.7187920, 28.5415306, -58.7118034, 28.5317841, -87.2505798, 87.2533340
42: -40.1411552, 24.4447803, -40.2943268, 24.4044666, -64.5456238, 64.7391052

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1745
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
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 747
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 900
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
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1675
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 548
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0604884
time: 79.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0647064, upper bound: 38.0686815
time: 102.58 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1739169
time: 81.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1820517
time: 103.30 seconds

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

Time for backsubstitution: 2.22 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0678078
time: 212.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0742994
time: 103.43 seconds

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

Time for backsubstitution: 2.25 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1137751
time: 76.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1156937
time: 104.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 183.35 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 37.9351437
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -38.0647064, upper bound: 37.9423853
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0438518
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0517390
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -38.0723779, upper bound: 37.9691714
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 37.9741269
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0105222
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0125246
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0034228
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0109895
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1116272
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1191533
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0059000
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -38.1851302, upper bound: 38.0110601
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0499484
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0516191
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 37.9926060
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0004540
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1065152
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1147878
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0311844
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0372138
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0745986
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0769717
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0604884
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -38.0647064, upper bound: 38.0686815
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1739169
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1820517
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0678078
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0742994
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1137751
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 183.35
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1156937

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -56.5506630, 42.5646362, -56.7017937, 42.7308846, -99.2815475, 99.2664337
1: -26.0538406, 34.9839859, -26.1596298, 35.1210136, -61.1748543, 61.1436157
2: -23.6825218, 36.7219086, -23.7734337, 36.8591461, -60.5416679, 60.4953423
3: -27.8905258, 40.9840240, -28.0164776, 41.1598969, -69.0504227, 69.0005035
4: -30.8392220, 40.9053040, -30.9482613, 41.0639038, -71.9031219, 71.8535614
5: -27.2831841, 42.2794571, -27.4045315, 42.4482536, -69.7314377, 69.6839905
6: -54.9766312, 26.8082504, -55.0315056, 26.8762970, -81.8529282, 81.8397522
7: -31.7784691, 40.1184998, -31.9057808, 40.3189087, -72.0973816, 72.0242767
8: -36.4499969, 49.0209274, -36.5350647, 49.1806793, -85.6306763, 85.5559921
9: -29.9905224, 37.8384895, -30.0807076, 37.8744736, -67.8649979, 67.9191971
10: -49.1580086, 47.1364899, -49.2894821, 47.1762238, -96.3342285, 96.4259720
11: -48.0893898, 28.2482147, -48.2690735, 28.3502445, -76.4396362, 76.5172882
12: -59.0836716, 30.0580406, -59.4759827, 30.1708336, -88.8203735, 89.0967560
13: -50.9438515, 46.5683517, -51.0785713, 46.6041832, -97.5480347, 97.6469269
14: -78.8246155, 41.7640686, -79.1523438, 41.7803040, -120.6049194, 120.9164124
15: -37.4157944, 34.9109650, -37.5411377, 34.9698601, -72.3856506, 72.4521027
16: -48.2028656, 36.4982338, -48.3182793, 36.6560669, -84.8589325, 84.8165131
17: -78.7803955, 33.2517853, -79.1913147, 33.3369255, -112.1173248, 112.4431000
18: -47.8798752, 32.9414558, -47.9781837, 33.0110741, -80.8909454, 80.9196396
19: -37.9729233, 18.9551964, -38.0710869, 19.0410767, -57.0139999, 57.0262833
20: -34.4067688, 24.6289158, -34.5316849, 24.6925392, -59.0993080, 59.1605988
21: -45.8265190, 24.3814659, -45.9681969, 24.4885483, -70.3150635, 70.3496628
22: -48.6480713, 24.7666359, -48.8050880, 24.8447590, -73.4928284, 73.5717239
23: -37.5696526, 26.0796623, -37.6736031, 26.1485748, -63.7182274, 63.7532654
24: -45.1466599, 28.6694508, -45.1543503, 28.7232494, -73.8699112, 73.8237991
25: -39.4032440, 29.0994644, -39.5048676, 29.1880646, -68.5913086, 68.6043320
26: -55.3742256, 37.8251953, -55.6789093, 37.9666443, -93.3408661, 93.5041046
27: -45.6340179, 29.9096222, -45.6396255, 29.9767513, -75.6107712, 75.5492477
28: -36.7134857, 29.6159477, -36.7700157, 29.6717873, -66.3852692, 66.3859634
29: -50.7421341, 24.0553379, -50.9341698, 24.1494942, -74.8916321, 74.9895096
30: -46.0720901, 32.9014969, -46.1466064, 33.0294647, -79.1015549, 79.0481033
31: -48.7679291, 27.4547348, -48.8590736, 27.5461273, -76.3140564, 76.3138123
32: -55.2811852, 24.1436024, -55.3844833, 24.1856022, -79.3363724, 79.3983459
33: -73.1788330, 31.4164162, -73.2313385, 31.5722466, -104.1320267, 103.9906693
34: -63.3261909, 17.5601330, -63.4152756, 17.6915035, -80.5552597, 80.4532166
35: -60.4234314, 24.1170902, -60.4621048, 24.2179623, -83.9735565, 83.8813477
36: -60.4772568, 25.0102272, -60.6238251, 25.0853844, -85.5599213, 85.6314240
37: -89.0553818, 18.1958237, -89.1211624, 18.2745934, -107.1391907, 107.1249695
38: -69.2491074, 28.7913666, -69.4396515, 28.8415718, -98.0906830, 98.2310181
39: -83.0622482, 30.4767189, -83.1124420, 30.6044559, -113.6667023, 113.5891571
40: -65.4745331, 21.1621361, -65.5181274, 21.3103256, -86.7405548, 86.6285019
41: -58.5037498, 28.3265572, -58.5455132, 28.4250431, -86.9287949, 86.8720703
42: -39.9191818, 24.1402397, -40.0150871, 24.1949883, -64.1141663, 64.1553268

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1691
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
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 737
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 537
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9110755, upper bound: 37.9332956
time: 102.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9110755, upper bound: 37.9332956
time: 105.08 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -56.7704010, 42.6879959, -56.7740631, 42.7449226, -99.5153198, 99.4620590
1: -26.1929436, 35.0624847, -26.2060356, 35.1303673, -61.3233109, 61.2685204
2: -23.8700714, 36.8191299, -23.8375092, 36.8697662, -60.7398376, 60.6566391
3: -28.1206341, 41.1190491, -28.0958900, 41.1766205, -69.2972565, 69.2149353
4: -31.0591927, 41.0230713, -31.0244236, 41.0775528, -72.1367493, 72.0474930
5: -27.4911156, 42.4204979, -27.4750938, 42.4640579, -69.9551697, 69.8955917
6: -55.0582962, 26.9295692, -55.0518112, 26.9149399, -81.9732361, 81.9813843
7: -31.9449635, 40.2130814, -31.9591064, 40.3316078, -72.2765732, 72.1721878
8: -36.6377640, 49.1286163, -36.5980644, 49.1943588, -85.8321228, 85.7266846
9: -30.1144085, 37.9262047, -30.1215553, 37.8982162, -68.0126266, 68.0477600
10: -49.2636986, 47.3006897, -49.3157921, 47.2270966, -96.4907990, 96.6164856
11: -48.2562332, 28.4516277, -48.2922173, 28.4160366, -76.6722717, 76.7438431
12: -59.2981110, 30.4132423, -59.4963913, 30.2923946, -89.1683655, 89.4813690
13: -51.0137177, 46.6886368, -51.0996056, 46.6398125, -97.6535339, 97.7882385
14: -79.0257645, 41.9819527, -79.1897278, 41.8569183, -120.8826828, 121.1716766
15: -37.6057281, 34.9813461, -37.5980225, 34.9889908, -72.5947189, 72.5793686
16: -48.3240738, 36.6149445, -48.3489685, 36.6902466, -85.0143204, 84.9639130
17: -79.0282974, 33.5406075, -79.2241821, 33.4360733, -112.4643707, 112.7647858
18: -47.9864807, 33.0591774, -48.0022583, 33.0499191, -81.0363998, 81.0614319
19: -38.0970306, 19.0764790, -38.0916252, 19.0837345, -57.1807632, 57.1681061
20: -34.5169067, 24.7542915, -34.5529976, 24.7360649, -59.2529716, 59.3072891
21: -45.9781761, 24.5462704, -45.9912567, 24.5459423, -70.5241165, 70.5375290
22: -48.7481613, 24.8921356, -48.8304024, 24.8841171, -73.6322784, 73.7225342
23: -37.6706772, 26.1650047, -37.6904755, 26.1775208, -63.8481979, 63.8554802
24: -45.2244263, 28.7166634, -45.1751785, 28.7374458, -73.9618683, 73.8918457
25: -39.4846001, 29.2090149, -39.5250587, 29.2246094, -68.7092133, 68.7340698
26: -55.5645485, 38.0885201, -55.7059746, 38.0587120, -93.6232605, 93.7944946
27: -45.7268448, 29.9553566, -45.6646652, 29.9910603, -75.7179031, 75.6200256
28: -36.8182526, 29.7235565, -36.7898026, 29.7088089, -66.5270615, 66.5133591
29: -50.8789139, 24.2163391, -50.9559593, 24.2042732, -75.0831909, 75.1723022
30: -46.1570587, 33.0242271, -46.1658058, 33.0706253, -79.2276840, 79.1900330
31: -48.8849220, 27.5455933, -48.8845482, 27.5754662, -76.4603882, 76.4301453
32: -55.3694725, 24.2865486, -55.4017372, 24.2335320, -79.4784088, 79.5619812
33: -73.3446045, 31.5309448, -73.2860565, 31.5960255, -104.3200226, 104.1676788
34: -63.4663506, 17.6673317, -63.4615402, 17.7109795, -80.7169189, 80.6234589
35: -60.5014076, 24.1770039, -60.4869537, 24.2311440, -84.0676880, 83.9756927
36: -60.5655518, 25.1418324, -60.6413956, 25.1280518, -85.6914749, 85.7808380
37: -89.1910172, 18.3332024, -89.1502838, 18.3194752, -107.3299866, 107.2966080
38: -69.3650742, 28.8846588, -69.4658051, 28.8685894, -98.2336655, 98.3504639
39: -83.1601944, 30.5433006, -83.1419373, 30.6186810, -113.7788773, 113.6852417
40: -65.5655212, 21.2289619, -65.5445099, 21.3270245, -86.8511810, 86.7245941
41: -58.5821266, 28.4120846, -58.5676308, 28.4525261, -87.0346527, 86.9797134
42: -40.0110703, 24.2642059, -40.0338211, 24.2352524, -64.2463226, 64.2980270

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 725
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
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 737
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0157599, upper bound: 37.9405264
time: 104.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0627059, upper bound: 37.9405264
time: 114.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -56.6250687, 42.5812836, -56.9225540, 42.8077621, -99.4328308, 99.5038376
1: -26.1066856, 34.9953003, -26.3154755, 35.1776276, -61.2843132, 61.3107758
2: -23.7719555, 36.7333336, -24.0346260, 36.9421539, -60.7141113, 60.7679596
3: -27.9900627, 41.0034332, -28.3068924, 41.2708740, -69.2609406, 69.3103256
4: -30.9307709, 40.9206924, -31.2166958, 41.1591454, -72.0899200, 72.1373901
5: -27.3795967, 42.2970886, -27.6872005, 42.5610924, -69.9406891, 69.9842911
6: -55.0023575, 26.8503265, -55.1113052, 26.9950123, -81.9973679, 81.9616318
7: -31.8626938, 40.1320763, -32.1536102, 40.3932266, -72.2559204, 72.2856903
8: -36.5338631, 49.0369034, -36.7822685, 49.2727203, -85.8065796, 85.8191681
9: -30.0173149, 37.8914337, -30.1638031, 38.0364609, -68.0537720, 68.0552368
10: -49.1885910, 47.2509956, -49.4286003, 47.5167923, -96.7053833, 96.6795959
11: -48.1164627, 28.3375759, -48.4077950, 28.5986710, -76.7151337, 76.7453690
12: -59.1053123, 30.2243462, -59.6424408, 30.6571388, -89.3307343, 89.4415054
13: -50.9763412, 46.6068611, -51.1673584, 46.7274246, -97.7037659, 97.7742157
14: -78.8692627, 41.8731766, -79.3376312, 42.0996552, -120.9689178, 121.2108078
15: -37.4810677, 34.9363785, -37.7219734, 35.0414009, -72.5224686, 72.6583557
16: -48.2412949, 36.5497437, -48.4466629, 36.8006287, -85.0419235, 84.9964066
17: -78.8112030, 33.3546219, -79.3531036, 33.6359253, -112.4471283, 112.7077255
18: -47.9093666, 33.0075455, -48.0932541, 33.2065697, -81.1159363, 81.1007996
19: -37.9963074, 18.9907475, -38.1729279, 19.1469269, -57.1432343, 57.1636734
20: -34.4323921, 24.6681309, -34.6322021, 24.8065605, -59.2389526, 59.3003311
21: -45.8518257, 24.4426537, -46.0917130, 24.6693916, -70.5212173, 70.5343628
22: -48.6742744, 24.8167763, -48.8887787, 24.9966965, -73.6709747, 73.7055511
23: -37.5893402, 26.1084824, -37.7579231, 26.2325325, -63.8218727, 63.8664055
24: -45.1749077, 28.6868935, -45.2422142, 28.7732010, -73.9481049, 73.9291077
25: -39.4246216, 29.1388969, -39.5713997, 29.3062496, -68.7308731, 68.7102966
26: -55.4046173, 37.9552078, -55.8303871, 38.3441391, -93.7487564, 93.7855988
27: -45.6701164, 29.9254227, -45.7567368, 30.0187073, -75.6888275, 75.6821594
28: -36.7372589, 29.6418018, -36.8614273, 29.7482338, -66.4854889, 66.5032272
29: -50.7652740, 24.1249657, -51.0248489, 24.3481846, -75.1134567, 75.1498108
30: -46.0933037, 32.9539337, -46.2216110, 33.1804390, -79.2737427, 79.1755447
31: -48.8006096, 27.4866486, -48.9756737, 27.6373119, -76.4379196, 76.4623260
32: -55.3032417, 24.2014923, -55.4768257, 24.3554268, -79.5284042, 79.5550690
33: -73.2420654, 31.4453316, -73.4194336, 31.6896000, -104.3227692, 104.2060699
34: -63.3579941, 17.5826454, -63.5131454, 17.7698212, -80.6813049, 80.5809402
35: -60.4588585, 24.1341114, -60.5667953, 24.2842503, -84.0895157, 84.0133820
36: -60.4966316, 25.0429649, -60.6850929, 25.1840153, -85.6786041, 85.7262115
37: -89.0885086, 18.2553120, -89.2480087, 18.4544106, -107.3553696, 107.3224411
38: -69.2840729, 28.8273735, -69.5540314, 28.9555588, -98.2396317, 98.3814087
39: -83.0952301, 30.4947567, -83.2136002, 30.6736622, -113.7688904, 113.7083588
40: -65.5072708, 21.1816711, -65.6234894, 21.3702908, -86.8374863, 86.7593536
41: -58.5299377, 28.3617058, -58.6295204, 28.5229950, -87.0529327, 86.9912262
42: -39.9391632, 24.2014370, -40.1096230, 24.3825645, -64.3217316, 64.3110580

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 765
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
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1691
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
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 618
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
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 971
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
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 737
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 548
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9110755, upper bound: 38.0417456
time: 171.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9110755, upper bound: 38.0417456
time: 139.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 312.81 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -37.9110755, upper bound: 37.9332956
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -37.9110755, upper bound: 37.9332956
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -38.0157599, upper bound: 37.9405264
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -38.0627059, upper bound: 37.9405264
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -37.9110755, upper bound: 38.0417456
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 312.81
Output dim: 2, lower bound: -37.9110755, upper bound: 38.0417456
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0517390
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -38.0723779, upper bound: 37.9691714
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 37.9741269
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0105222
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0125246
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0034228
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0109895
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1116272
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1191533
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0059000
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -38.1851302, upper bound: 38.0110601
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0499484
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0516191
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9607965, upper bound: 37.9926060
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9607965, upper bound: 38.0004540
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1065152
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9607965, upper bound: 38.1147878
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0311844
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0372138
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0745986
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0769717
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.0604884
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -38.0647064, upper bound: 38.0686815
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1739169
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9611364, upper bound: 38.1820517
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0678078
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.0742994
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1137751
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 312.81
Output dim: 2, lower bound: -37.9668060, upper bound: 38.1156937

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 164.85 + 7220.14 = 7384.99 seconds
