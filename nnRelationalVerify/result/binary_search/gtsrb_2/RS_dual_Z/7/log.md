## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 97.2844837351
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716)
1: (-70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341)
2: (-63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282)
3: (-72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957)
4: (-76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267)
5: (-68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710)
6: (-102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202)
7: (-84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920)
8: (-89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154)
9: (-78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873)
10: (-111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498)
11: (-111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485)
12: (-111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295)
13: (-110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117)
14: (-163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874)
15: (-92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756)
16: (-118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149)
17: (-164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765)
18: (-102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608)
19: (-85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756)
20: (-74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135)
21: (-104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721)
22: (-113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161)
23: (-86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248)
24: (-103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021)
25: (-91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324)
26: (-122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165)
27: (-104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583)
28: (-85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112)
29: (-119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958)
30: (-102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372)
31: (-106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931)
32: (-100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959)
33: (-141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360)
34: (-120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018)
35: (-120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321)
36: (-117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379)
37: (-164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464)
38: (-145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569)
39: (-168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181)
40: (-135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575)
41: (-100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632)
42: (-75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712)

## BASE Result
execution time: IAR + LP analysis = 2.94 + 160.17 = 163.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -107.9309805, upper bound: 107.9309806


# Binary Search by BASE starts (time budget: 17836.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=159.03338623046875
rel_dist={5: [-102.21084835141579, 102.21084836698037]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=159.03338623046875
rel_dist={5: [-97.30393826817422, 97.30393825827855]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=159.03338623046875
rel_dist={5: [-92.81938600404467, 92.81938600410132]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=159.03338623046875
rel_dist={5: [-95.19916083299321, 95.19916083562723]}

## Binary Search Result
Binary search time: 696.13 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 17140.76 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4376365, upper bound: 103.3813566
time: 138.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3813565, upper bound: 103.4376365
time: 94.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 233.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 233.41
Output dim: 5, lower bound: -103.4376365, upper bound: 103.3813566
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 233.41
Output dim: 5, lower bound: -103.3813565, upper bound: 103.4376365

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4025372, upper bound: 103.3744381
time: 100.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4304005, upper bound: 103.3387384
time: 115.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3387384, upper bound: 103.4304005
time: 111.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3744381, upper bound: 103.4025372
time: 114.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 228.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 228.57
Output dim: 5, lower bound: -103.4025372, upper bound: 103.3744381
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 228.57
Output dim: 5, lower bound: -103.4304005, upper bound: 103.3387384
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 228.57
Output dim: 5, lower bound: -103.3387384, upper bound: 103.4304005
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 228.57
Output dim: 5, lower bound: -103.3744381, upper bound: 103.4025372

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3415059, upper bound: 103.3726614
time: 123.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3984706, upper bound: 103.3032986
time: 111.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3599350, upper bound: 103.3343413
time: 177.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4286735, upper bound: 103.2748783
time: 90.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2748783, upper bound: 103.4286735
time: 189.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3343412, upper bound: 103.3599350
time: 183.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3032986, upper bound: 103.3984707
time: 125.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3726614, upper bound: 103.3415060
time: 120.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 248.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3415059, upper bound: 103.3726614
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3984706, upper bound: 103.3032986
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3599350, upper bound: 103.3343413
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.4286735, upper bound: 103.2748783
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.2748783, upper bound: 103.4286735
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3343412, upper bound: 103.3599350
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3032986, upper bound: 103.3984707
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 248.31
Output dim: 5, lower bound: -103.3726614, upper bound: 103.3415060

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2591445, upper bound: 103.3681533
time: 127.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3373850, upper bound: 103.2969335
time: 151.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3208309, upper bound: 103.2989219
time: 116.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3941614, upper bound: 103.2224799
time: 115.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2810693, upper bound: 103.3304456
time: 115.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2810693, upper bound: 103.2521188
time: 177.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3534772, upper bound: 103.2713707
time: 137.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.4243041, upper bound: 103.1889994
time: 271.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.1889994, upper bound: 103.4243041
time: 103.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2713707, upper bound: 103.3534772
time: 138.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2521188, upper bound: 103.3554519
time: 110.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.3304455, upper bound: 103.2810693
time: 215.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2224799, upper bound: 103.3941614
time: 153.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -103.2989219, upper bound: 103.3208309
time: 233.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 390.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2591445, upper bound: 103.3681533
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.3373850, upper bound: 103.2969335
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.3208309, upper bound: 103.2989219
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.3941614, upper bound: 103.2224799
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2810693, upper bound: 103.3304456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2810693, upper bound: 103.2521188
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.3534772, upper bound: 103.2713707
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.4243041, upper bound: 103.1889994
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.1889994, upper bound: 103.4243041
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2713707, upper bound: 103.3534772
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2521188, upper bound: 103.3554519
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.3304455, upper bound: 103.2810693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2224799, upper bound: 103.3941614
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 390.06
Output dim: 5, lower bound: -103.2989219, upper bound: 103.3208309
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 390.06
Output dim: 5, lower bound: -103.3726614, upper bound: 103.3415060
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=159.03338623046875
rel_dist={5: [-103.46050891932094, 103.46050895689399]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.1244609, upper bound: 99.0875919
time: 135.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0875920, upper bound: 99.1244609
time: 101.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 237.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 237.50
Output dim: 5, lower bound: -99.1244609, upper bound: 99.0875919
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 237.50
Output dim: 5, lower bound: -99.0875920, upper bound: 99.1244609

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0852942, upper bound: 99.0820219
time: 103.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.1183406, upper bound: 99.0412284
time: 299.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0412284, upper bound: 99.1183406
time: 135.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0820220, upper bound: 99.0852942
time: 122.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 260.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 260.47
Output dim: 5, lower bound: -99.0852942, upper bound: 99.0820219
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 260.47
Output dim: 5, lower bound: -99.1183406, upper bound: 99.0412284
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 260.47
Output dim: 5, lower bound: -99.0412284, upper bound: 99.1183406
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 260.47
Output dim: 5, lower bound: -99.0820220, upper bound: 99.0852942

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0331198, upper bound: 99.0802959
time: 147.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0813796, upper bound: 99.0179552
time: 141.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0548303, upper bound: 99.0372375
time: 109.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.1166250, upper bound: 98.9893756
time: 127.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9893756, upper bound: 99.1166250
time: 145.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0372375, upper bound: 99.0548303
time: 136.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0179552, upper bound: 99.0813796
time: 106.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0802959, upper bound: 99.0331198
time: 119.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 229.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0331198, upper bound: 99.0802959
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0813796, upper bound: 99.0179552
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0548303, upper bound: 99.0372375
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.1166250, upper bound: 98.9893756
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -98.9893756, upper bound: 99.1166250
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0372375, upper bound: 99.0548303
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0179552, upper bound: 99.0813796
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.26
Output dim: 5, lower bound: -99.0802959, upper bound: 99.0331198

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9650061, upper bound: 99.0780159
time: 91.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9650061, upper bound: 99.0157721
time: 167.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0141413, upper bound: 99.0153416
time: 148.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0141413, upper bound: 98.9504768
time: 110.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9881557, upper bound: 99.0348307
time: 104.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0523004, upper bound: 98.9688735
time: 110.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0527705, upper bound: 98.9864065
time: 126.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.1143512, upper bound: 98.9198731
time: 195.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9198731, upper bound: 99.1143512
time: 117.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9198731, upper bound: 99.0527705
time: 125.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9688736, upper bound: 99.0523004
time: 375.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9688736, upper bound: 98.9881557
time: 106.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.9504768, upper bound: 99.0789897
time: 96.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -99.0153416, upper bound: 99.0141413
time: 119.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 217.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9650061, upper bound: 99.0780159
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9650061, upper bound: 99.0157721
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.0141413, upper bound: 99.0153416
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.0141413, upper bound: 98.9504768
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9881557, upper bound: 99.0348307
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.0523004, upper bound: 98.9688735
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.0527705, upper bound: 98.9864065
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.1143512, upper bound: 98.9198731
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9198731, upper bound: 99.1143512
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9198731, upper bound: 99.0527705
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9688736, upper bound: 99.0523004
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9688736, upper bound: 98.9881557
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -98.9504768, upper bound: 99.0789897
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 217.86
Output dim: 5, lower bound: -99.0153416, upper bound: 99.0141413
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 217.86
Output dim: 5, lower bound: -99.0802959, upper bound: 99.0331198
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=159.03338623046875
rel_dist={5: [-99.13839545242836, 99.13839544583911]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2931018, upper bound: 97.2646353
time: 105.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2646353, upper bound: 97.2931018
time: 115.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 221.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 221.99
Output dim: 5, lower bound: -97.2931018, upper bound: 97.2646353
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 221.99
Output dim: 5, lower bound: -97.2646353, upper bound: 97.2931018

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2576601, upper bound: 97.2594205
time: 146.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2880378, upper bound: 97.2232472
time: 102.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2232472, upper bound: 97.2880378
time: 227.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2594205, upper bound: 97.2576601
time: 119.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 349.51 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 349.51
Output dim: 5, lower bound: -97.2576601, upper bound: 97.2594205
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 349.51
Output dim: 5, lower bound: -97.2880378, upper bound: 97.2232472
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 349.51
Output dim: 5, lower bound: -97.2232472, upper bound: 97.2880378
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 349.51
Output dim: 5, lower bound: -97.2594205, upper bound: 97.2576601

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2343011, upper bound: 97.2198792
time: 132.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2865414, upper bound: 97.1811542
time: 112.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1811543, upper bound: 97.2865414
time: 127.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2198793, upper bound: 97.2343010
time: 95.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 224.95 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 224.95
Output dim: 5, lower bound: -97.2343011, upper bound: 97.2198792
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 224.95
Output dim: 5, lower bound: -97.2865414, upper bound: 97.1811542
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 224.95
Output dim: 5, lower bound: -97.1811543, upper bound: 97.2865414
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 224.95
Output dim: 5, lower bound: -97.2198793, upper bound: 97.2343010

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2298380, upper bound: 97.1773788
time: 161.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.2298380, upper bound: 97.1231992
time: 119.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 660

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1231992, upper bound: 97.2835109
time: 105.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1231992, upper bound: 97.2298379
time: 133.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 240.86 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 240.86
Output dim: 5, lower bound: -97.2298380, upper bound: 97.1773788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 240.86
Output dim: 5, lower bound: -97.2298380, upper bound: 97.1231992
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 240.86
Output dim: 5, lower bound: -97.1231992, upper bound: 97.2835109
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 240.86
Output dim: 5, lower bound: -97.1231992, upper bound: 97.2298379
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=159.03338623046875
rel_dist={5: [-97.30393826817422, 97.30393825827855]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 10125.27 seconds
