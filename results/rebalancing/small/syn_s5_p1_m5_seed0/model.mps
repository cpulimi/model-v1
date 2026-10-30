* Signature: 0x83962e1cea54ada8
NAME rebalancing_base
ROWS
 N  OBJ
 L  c2_excess[s2,p0]
 L  c2_excess[s4,p0]
 E  c3_need[s3,p0]
 E  c3_need[s0,p0]
 E  c3_need[s1,p0]
 L  c4_ship_link[s2]
 L  c4_ship_link[s4]
 L  c5_recv_link[s0]
 L  c5_recv_link[s1]
 L  c5_recv_link[s3]
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    X[s2,s3,p0]  OBJ       132
    X[s2,s3,p0]  c2_excess[s2,p0]  1
    X[s2,s3,p0]  c3_need[s3,p0]  1
    X[s2,s3,p0]  c4_ship_link[s2]  1
    X[s2,s3,p0]  c5_recv_link[s3]  1
    X[s2,s0,p0]  OBJ       144
    X[s2,s0,p0]  c2_excess[s2,p0]  1
    X[s2,s0,p0]  c3_need[s0,p0]  1
    X[s2,s0,p0]  c4_ship_link[s2]  1
    X[s2,s0,p0]  c5_recv_link[s0]  1
    X[s2,s1,p0]  OBJ       78
    X[s2,s1,p0]  c2_excess[s2,p0]  1
    X[s2,s1,p0]  c3_need[s1,p0]  1
    X[s2,s1,p0]  c4_ship_link[s2]  1
    X[s2,s1,p0]  c5_recv_link[s1]  1
    X[s4,s3,p0]  OBJ       135
    X[s4,s3,p0]  c2_excess[s4,p0]  1
    X[s4,s3,p0]  c3_need[s3,p0]  1
    X[s4,s3,p0]  c4_ship_link[s4]  1
    X[s4,s3,p0]  c5_recv_link[s3]  1
    X[s4,s0,p0]  OBJ       53
    X[s4,s0,p0]  c2_excess[s4,p0]  1
    X[s4,s0,p0]  c3_need[s0,p0]  1
    X[s4,s0,p0]  c4_ship_link[s4]  1
    X[s4,s0,p0]  c5_recv_link[s0]  1
    X[s4,s1,p0]  OBJ       127
    X[s4,s1,p0]  c2_excess[s4,p0]  1
    X[s4,s1,p0]  c3_need[s1,p0]  1
    X[s4,s1,p0]  c4_ship_link[s4]  1
    X[s4,s1,p0]  c5_recv_link[s1]  1
    T[m0,s3,p0]  OBJ       92
    T[m0,s3,p0]  c3_need[s3,p0]  1
    T[m0,s3,p0]  c5_recv_link[s3]  1
    T[m1,s3,p0]  OBJ       84
    T[m1,s3,p0]  c3_need[s3,p0]  1
    T[m1,s3,p0]  c5_recv_link[s3]  1
    T[m2,s3,p0]  OBJ       162
    T[m2,s3,p0]  c3_need[s3,p0]  1
    T[m2,s3,p0]  c5_recv_link[s3]  1
    T[m3,s3,p0]  OBJ       77
    T[m3,s3,p0]  c3_need[s3,p0]  1
    T[m3,s3,p0]  c5_recv_link[s3]  1
    T[m4,s3,p0]  OBJ       129
    T[m4,s3,p0]  c3_need[s3,p0]  1
    T[m4,s3,p0]  c5_recv_link[s3]  1
    T[m0,s0,p0]  OBJ       83
    T[m0,s0,p0]  c3_need[s0,p0]  1
    T[m0,s0,p0]  c5_recv_link[s0]  1
    T[m1,s0,p0]  OBJ       105
    T[m1,s0,p0]  c3_need[s0,p0]  1
    T[m1,s0,p0]  c5_recv_link[s0]  1
    T[m2,s0,p0]  OBJ       123
    T[m2,s0,p0]  c3_need[s0,p0]  1
    T[m2,s0,p0]  c5_recv_link[s0]  1
    T[m3,s0,p0]  OBJ       117
    T[m3,s0,p0]  c3_need[s0,p0]  1
    T[m3,s0,p0]  c5_recv_link[s0]  1
    T[m4,s0,p0]  OBJ       115
    T[m4,s0,p0]  c3_need[s0,p0]  1
    T[m4,s0,p0]  c5_recv_link[s0]  1
    T[m0,s1,p0]  OBJ       77
    T[m0,s1,p0]  c3_need[s1,p0]  1
    T[m0,s1,p0]  c5_recv_link[s1]  1
    T[m1,s1,p0]  OBJ       75
    T[m1,s1,p0]  c3_need[s1,p0]  1
    T[m1,s1,p0]  c5_recv_link[s1]  1
    T[m2,s1,p0]  OBJ       87
    T[m2,s1,p0]  c3_need[s1,p0]  1
    T[m2,s1,p0]  c5_recv_link[s1]  1
    T[m3,s1,p0]  OBJ       75
    T[m3,s1,p0]  c3_need[s1,p0]  1
    T[m3,s1,p0]  c5_recv_link[s1]  1
    T[m4,s1,p0]  OBJ       142
    T[m4,s1,p0]  c3_need[s1,p0]  1
    T[m4,s1,p0]  c5_recv_link[s1]  1
    U[s3,p0]  OBJ       10000
    U[s3,p0]  c3_need[s3,p0]  1
    U[s0,p0]  OBJ       10000
    U[s0,p0]  c3_need[s0,p0]  1
    U[s1,p0]  OBJ       10000
    U[s1,p0]  c3_need[s1,p0]  1
    Y[s2]     OBJ       100
    Y[s2]     c4_ship_link[s2]  -27
    Y[s4]     OBJ       100
    Y[s4]     c4_ship_link[s4]  -21
    Z[s0]     OBJ       1000
    Z[s0]     c5_recv_link[s0]  -132
    Z[s1]     OBJ       1000
    Z[s1]     c5_recv_link[s1]  -115
    Z[s3]     OBJ       1000
    Z[s3]     c5_recv_link[s3]  -67
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      c2_excess[s2,p0]  27
    RHS1      c2_excess[s4,p0]  21
    RHS1      c3_need[s3,p0]  67
    RHS1      c3_need[s0,p0]  132
    RHS1      c3_need[s1,p0]  115
BOUNDS
 LI BND1      X[s2,s3,p0]  0
 LI BND1      X[s2,s0,p0]  0
 LI BND1      X[s2,s1,p0]  0
 LI BND1      X[s4,s3,p0]  0
 LI BND1      X[s4,s0,p0]  0
 LI BND1      X[s4,s1,p0]  0
 LI BND1      T[m0,s3,p0]  0
 LI BND1      T[m1,s3,p0]  0
 LI BND1      T[m2,s3,p0]  0
 LI BND1      T[m3,s3,p0]  0
 LI BND1      T[m4,s3,p0]  0
 LI BND1      T[m0,s0,p0]  0
 LI BND1      T[m1,s0,p0]  0
 LI BND1      T[m2,s0,p0]  0
 LI BND1      T[m3,s0,p0]  0
 LI BND1      T[m4,s0,p0]  0
 LI BND1      T[m0,s1,p0]  0
 LI BND1      T[m1,s1,p0]  0
 LI BND1      T[m2,s1,p0]  0
 LI BND1      T[m3,s1,p0]  0
 LI BND1      T[m4,s1,p0]  0
 LI BND1      U[s3,p0]  0
 LI BND1      U[s0,p0]  0
 LI BND1      U[s1,p0]  0
 BV BND1      Y[s2]   
 BV BND1      Y[s4]   
 BV BND1      Z[s0]   
 BV BND1      Z[s1]   
 BV BND1      Z[s3]   
ENDATA
