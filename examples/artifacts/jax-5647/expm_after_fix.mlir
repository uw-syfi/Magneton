module @jit_expm_after_fix attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256xf32>) -> (tensor<256x256xf32> {jax.result_info = ""}) {
    %0 = call @expm_after_fix(%arg0) : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @expm_after_fix(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<[0.0149558522, 0.253939837, 0.950417876, 2.09784794]> : tensor<4xf32>
    %0 = stablehlo.abs %arg0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst_1) applies stablehlo.maximum across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %3 = call @digitize(%2, %cst) : (tensor<f32>, tensor<4xf32>) -> tensor<i32>
    %c = stablehlo.constant dense<4> : tensor<i32>
    %4 = stablehlo.compare  GE, %3, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %cst_2 = stablehlo.constant dense<5.37192059> : tensor<f32>
    %5 = stablehlo.divide %2, %cst_2 : tensor<f32>
    %6 = stablehlo.log %5 : tensor<f32>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %7 = stablehlo.log %cst_3 : tensor<f32>
    %8 = stablehlo.divide %6, %7 : tensor<f32>
    %9 = stablehlo.floor %8 : tensor<f32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.maximum %cst_4, %9 : tensor<f32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = call @_where_0(%4, %10, %cst_5) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.convert %11 : (tensor<f32>) -> tensor<i32>
    %13 = stablehlo.compare  GE, %3, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %cst_6 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %14 = stablehlo.convert %12 : (tensor<i32>) -> tensor<f32>
    %15 = stablehlo.power %cst_6, %14 : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %17 = stablehlo.divide %arg0, %16 : tensor<256x256xf32>
    %18 = call @_where_1(%13, %17, %arg0) : (tensor<i1>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.clamp %c_7, %3, %c : tensor<i32>
    %20:2 = "stablehlo.case"(%19) ({
      %26 = stablehlo.iota dim = 0 : tensor<256x256xi32>
      %27 = stablehlo.iota dim = 1 : tensor<256x256xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
      %29 = stablehlo.add %26, %28 : tensor<256x256xi32>
      %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
      %31 = stablehlo.convert %30 : (tensor<256x256xi1>) -> tensor<256x256xf32>
      %32 = stablehlo.dot_general %18, %18, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %33 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %34 = stablehlo.multiply %33, %32 : tensor<256x256xf32>
      %cst_11 = stablehlo.constant dense<6.000000e+01> : tensor<f32>
      %35 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %36 = stablehlo.multiply %35, %31 : tensor<256x256xf32>
      %37 = stablehlo.add %34, %36 : tensor<256x256xf32>
      %38 = stablehlo.dot_general %18, %37, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_12 = stablehlo.constant dense<1.200000e+01> : tensor<f32>
      %39 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %40 = stablehlo.multiply %39, %32 : tensor<256x256xf32>
      %cst_13 = stablehlo.constant dense<1.200000e+02> : tensor<f32>
      %41 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %42 = stablehlo.multiply %41, %31 : tensor<256x256xf32>
      %43 = stablehlo.add %40, %42 : tensor<256x256xf32>
      stablehlo.return %38, %43 : tensor<256x256xf32>, tensor<256x256xf32>
    }, {
      %26 = stablehlo.iota dim = 0 : tensor<256x256xi32>
      %27 = stablehlo.iota dim = 1 : tensor<256x256xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
      %29 = stablehlo.add %26, %28 : tensor<256x256xi32>
      %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
      %31 = stablehlo.convert %30 : (tensor<256x256xi1>) -> tensor<256x256xf32>
      %32 = stablehlo.dot_general %18, %18, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %33 = stablehlo.dot_general %32, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %34 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %35 = stablehlo.multiply %34, %33 : tensor<256x256xf32>
      %cst_11 = stablehlo.constant dense<4.200000e+02> : tensor<f32>
      %36 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %37 = stablehlo.multiply %36, %32 : tensor<256x256xf32>
      %38 = stablehlo.add %35, %37 : tensor<256x256xf32>
      %cst_12 = stablehlo.constant dense<1.512000e+04> : tensor<f32>
      %39 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %40 = stablehlo.multiply %39, %31 : tensor<256x256xf32>
      %41 = stablehlo.add %38, %40 : tensor<256x256xf32>
      %42 = stablehlo.dot_general %18, %41, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_13 = stablehlo.constant dense<3.000000e+01> : tensor<f32>
      %43 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %44 = stablehlo.multiply %43, %33 : tensor<256x256xf32>
      %cst_14 = stablehlo.constant dense<3.360000e+03> : tensor<f32>
      %45 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %46 = stablehlo.multiply %45, %32 : tensor<256x256xf32>
      %47 = stablehlo.add %44, %46 : tensor<256x256xf32>
      %cst_15 = stablehlo.constant dense<3.024000e+04> : tensor<f32>
      %48 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %49 = stablehlo.multiply %48, %31 : tensor<256x256xf32>
      %50 = stablehlo.add %47, %49 : tensor<256x256xf32>
      stablehlo.return %42, %50 : tensor<256x256xf32>, tensor<256x256xf32>
    }, {
      %26 = stablehlo.iota dim = 0 : tensor<256x256xi32>
      %27 = stablehlo.iota dim = 1 : tensor<256x256xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
      %29 = stablehlo.add %26, %28 : tensor<256x256xi32>
      %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
      %31 = stablehlo.convert %30 : (tensor<256x256xi1>) -> tensor<256x256xf32>
      %32 = stablehlo.dot_general %18, %18, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %33 = stablehlo.dot_general %32, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %34 = stablehlo.dot_general %33, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %35 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %36 = stablehlo.multiply %35, %34 : tensor<256x256xf32>
      %cst_11 = stablehlo.constant dense<1.512000e+03> : tensor<f32>
      %37 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %38 = stablehlo.multiply %37, %33 : tensor<256x256xf32>
      %39 = stablehlo.add %36, %38 : tensor<256x256xf32>
      %cst_12 = stablehlo.constant dense<2.772000e+05> : tensor<f32>
      %40 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %41 = stablehlo.multiply %40, %32 : tensor<256x256xf32>
      %42 = stablehlo.add %39, %41 : tensor<256x256xf32>
      %cst_13 = stablehlo.constant dense<8.648640e+06> : tensor<f32>
      %43 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %44 = stablehlo.multiply %43, %31 : tensor<256x256xf32>
      %45 = stablehlo.add %42, %44 : tensor<256x256xf32>
      %46 = stablehlo.dot_general %18, %45, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_14 = stablehlo.constant dense<5.600000e+01> : tensor<f32>
      %47 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %48 = stablehlo.multiply %47, %34 : tensor<256x256xf32>
      %cst_15 = stablehlo.constant dense<2.520000e+04> : tensor<f32>
      %49 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %50 = stablehlo.multiply %49, %33 : tensor<256x256xf32>
      %51 = stablehlo.add %48, %50 : tensor<256x256xf32>
      %cst_16 = stablehlo.constant dense<1.995840e+06> : tensor<f32>
      %52 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %53 = stablehlo.multiply %52, %32 : tensor<256x256xf32>
      %54 = stablehlo.add %51, %53 : tensor<256x256xf32>
      %cst_17 = stablehlo.constant dense<0x4B83F7C0> : tensor<f32>
      %55 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %56 = stablehlo.multiply %55, %31 : tensor<256x256xf32>
      %57 = stablehlo.add %54, %56 : tensor<256x256xf32>
      stablehlo.return %46, %57 : tensor<256x256xf32>, tensor<256x256xf32>
    }, {
      %26 = stablehlo.iota dim = 0 : tensor<256x256xi32>
      %27 = stablehlo.iota dim = 1 : tensor<256x256xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
      %29 = stablehlo.add %26, %28 : tensor<256x256xi32>
      %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
      %31 = stablehlo.convert %30 : (tensor<256x256xi1>) -> tensor<256x256xf32>
      %32 = stablehlo.dot_general %18, %18, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %33 = stablehlo.dot_general %32, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %34 = stablehlo.dot_general %33, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %35 = stablehlo.dot_general %34, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %37 = stablehlo.multiply %36, %35 : tensor<256x256xf32>
      %cst_11 = stablehlo.constant dense<3.960000e+03> : tensor<f32>
      %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %39 = stablehlo.multiply %38, %34 : tensor<256x256xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x256xf32>
      %cst_12 = stablehlo.constant dense<2.162160e+06> : tensor<f32>
      %41 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %42 = stablehlo.multiply %41, %33 : tensor<256x256xf32>
      %43 = stablehlo.add %40, %42 : tensor<256x256xf32>
      %cst_13 = stablehlo.constant dense<0x4D9056FA> : tensor<f32>
      %44 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %45 = stablehlo.multiply %44, %32 : tensor<256x256xf32>
      %46 = stablehlo.add %43, %45 : tensor<256x256xf32>
      %cst_14 = stablehlo.constant dense<8.82161254E+9> : tensor<f32>
      %47 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %48 = stablehlo.multiply %47, %31 : tensor<256x256xf32>
      %49 = stablehlo.add %46, %48 : tensor<256x256xf32>
      %50 = stablehlo.dot_general %18, %49, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_15 = stablehlo.constant dense<9.000000e+01> : tensor<f32>
      %51 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %52 = stablehlo.multiply %51, %35 : tensor<256x256xf32>
      %cst_16 = stablehlo.constant dense<1.108800e+05> : tensor<f32>
      %53 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %54 = stablehlo.multiply %53, %34 : tensor<256x256xf32>
      %55 = stablehlo.add %52, %54 : tensor<256x256xf32>
      %cst_17 = stablehlo.constant dense<0x4BE6F190> : tensor<f32>
      %56 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %57 = stablehlo.multiply %56, %33 : tensor<256x256xf32>
      %58 = stablehlo.add %55, %57 : tensor<256x256xf32>
      %cst_18 = stablehlo.constant dense<2.0756736E+9> : tensor<f32>
      %59 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %60 = stablehlo.multiply %59, %32 : tensor<256x256xf32>
      %61 = stablehlo.add %58, %60 : tensor<256x256xf32>
      %cst_19 = stablehlo.constant dense<1.76432251E+10> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %63 = stablehlo.multiply %62, %31 : tensor<256x256xf32>
      %64 = stablehlo.add %61, %63 : tensor<256x256xf32>
      stablehlo.return %50, %64 : tensor<256x256xf32>, tensor<256x256xf32>
    }, {
      %26 = stablehlo.iota dim = 0 : tensor<256x256xi32>
      %27 = stablehlo.iota dim = 1 : tensor<256x256xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
      %29 = stablehlo.add %26, %28 : tensor<256x256xi32>
      %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
      %31 = stablehlo.convert %30 : (tensor<256x256xi1>) -> tensor<256x256xf32>
      %32 = stablehlo.dot_general %18, %18, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %33 = stablehlo.dot_general %32, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %34 = stablehlo.dot_general %33, %32, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %35 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %36 = stablehlo.multiply %35, %34 : tensor<256x256xf32>
      %cst_11 = stablehlo.constant dense<1.638000e+04> : tensor<f32>
      %37 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %38 = stablehlo.multiply %37, %33 : tensor<256x256xf32>
      %39 = stablehlo.add %36, %38 : tensor<256x256xf32>
      %cst_12 = stablehlo.constant dense<4.084080e+07> : tensor<f32>
      %40 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %41 = stablehlo.multiply %40, %32 : tensor<256x256xf32>
      %42 = stablehlo.add %39, %41 : tensor<256x256xf32>
      %43 = stablehlo.dot_general %34, %42, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_13 = stablehlo.constant dense<3.35221289E+10> : tensor<f32>
      %44 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %45 = stablehlo.multiply %44, %34 : tensor<256x256xf32>
      %46 = stablehlo.add %43, %45 : tensor<256x256xf32>
      %cst_14 = stablehlo.constant dense<1.05594707E+13> : tensor<f32>
      %47 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %48 = stablehlo.multiply %47, %33 : tensor<256x256xf32>
      %49 = stablehlo.add %46, %48 : tensor<256x256xf32>
      %cst_15 = stablehlo.constant dense<1.18735378E+15> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %51 = stablehlo.multiply %50, %32 : tensor<256x256xf32>
      %52 = stablehlo.add %49, %51 : tensor<256x256xf32>
      %cst_16 = stablehlo.constant dense<3.23823762E+16> : tensor<f32>
      %53 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %54 = stablehlo.multiply %53, %31 : tensor<256x256xf32>
      %55 = stablehlo.add %52, %54 : tensor<256x256xf32>
      %56 = stablehlo.dot_general %18, %55, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_17 = stablehlo.constant dense<1.820000e+02> : tensor<f32>
      %57 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %58 = stablehlo.multiply %57, %34 : tensor<256x256xf32>
      %cst_18 = stablehlo.constant dense<9.609600e+05> : tensor<f32>
      %59 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %60 = stablehlo.multiply %59, %33 : tensor<256x256xf32>
      %61 = stablehlo.add %58, %60 : tensor<256x256xf32>
      %cst_19 = stablehlo.constant dense<1.32324198E+9> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %63 = stablehlo.multiply %62, %32 : tensor<256x256xf32>
      %64 = stablehlo.add %61, %63 : tensor<256x256xf32>
      %65 = stablehlo.dot_general %34, %64, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      %cst_20 = stablehlo.constant dense<6.70442586E+11> : tensor<f32>
      %66 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %67 = stablehlo.multiply %66, %34 : tensor<256x256xf32>
      %68 = stablehlo.add %65, %67 : tensor<256x256xf32>
      %cst_21 = stablehlo.constant dense<1.29060194E+14> : tensor<f32>
      %69 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %70 = stablehlo.multiply %69, %33 : tensor<256x256xf32>
      %71 = stablehlo.add %68, %70 : tensor<256x256xf32>
      %cst_22 = stablehlo.constant dense<7.771770e+15> : tensor<f32>
      %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %73 = stablehlo.multiply %72, %32 : tensor<256x256xf32>
      %74 = stablehlo.add %71, %73 : tensor<256x256xf32>
      %cst_23 = stablehlo.constant dense<6.47647525E+16> : tensor<f32>
      %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %76 = stablehlo.multiply %75, %31 : tensor<256x256xf32>
      %77 = stablehlo.add %74, %76 : tensor<256x256xf32>
      stablehlo.return %56, %77 : tensor<256x256xf32>, tensor<256x256xf32>
    }) : (tensor<i32>) -> (tensor<256x256xf32>, tensor<256x256xf32>)
    %21 = stablehlo.add %20#0, %20#1 : tensor<256x256xf32>
    %22 = stablehlo.negate %20#0 : tensor<256x256xf32>
    %23 = stablehlo.add %22, %20#1 : tensor<256x256xf32>
    %24 = call @solve(%23, %21) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %25:3 = stablehlo.while(%iterArg = %c_8, %iterArg_9 = %12, %iterArg_10 = %24) : tensor<i32>, tensor<i32>, tensor<256x256xf32>
     cond {
      %26 = stablehlo.compare  LT, %iterArg, %iterArg_9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %26 : tensor<i1>
    } do {
      %c_11 = stablehlo.constant dense<1> : tensor<i32>
      %26 = stablehlo.add %iterArg, %c_11 : tensor<i32>
      %27 = stablehlo.dot_general %iterArg_10, %iterArg_10, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      stablehlo.return %26, %iterArg_9, %27 : tensor<i32>, tensor<i32>, tensor<256x256xf32>
    }
    return %25#2 : tensor<256x256xf32>
  }
  func.func private @digitize(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> tensor<i32> {
    %c = stablehlo.constant dense<3> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg1, %c, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.compare  GE, %1, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = call @searchsorted(%arg1, %arg0) : (tensor<4xf32>, tensor<f32>) -> tensor<i32>
    %6 = stablehlo.reverse %arg1, dims = [0] : tensor<4xf32>
    %7 = call @searchsorted(%6, %arg0) : (tensor<4xf32>, tensor<f32>) -> tensor<i32>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.subtract %c_0, %7 : tensor<i32>
    %9 = call @_where(%4, %5, %8) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %9 : tensor<i32>
  }
  func.func private @searchsorted(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:5 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c_1, %iterArg_4 = %c, %iterArg_5 = %c_0) : tensor<4xf32>, tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>
     cond {
      %c_6 = stablehlo.constant dense<3> : tensor<i32>
      %1 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1:2 = func.call @None(%iterArg, %iterArg_2, %iterArg_4, %iterArg_5) : (tensor<4xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i32>
      %2 = stablehlo.add %iterArg_3, %c_6 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_2, %2, %1#0, %1#1 : tensor<4xf32>, tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>
    }
    return %0#4 : tensor<i32>
  }
  func.func private @None(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<ui32>
    %1 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<ui32>
    %2 = stablehlo.add %0, %1 : tensor<ui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %3 = stablehlo.divide %2, %c : tensor<ui32>
    %4 = stablehlo.convert %3 : (tensor<ui32>) -> tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_1 = stablehlo.constant dense<4> : tensor<i32>
    %6 = stablehlo.add %4, %c_1 : tensor<i32>
    %7 = stablehlo.select %5, %6, %4 : tensor<i1>, tensor<i32>
    %8 = stablehlo.dynamic_slice %arg0, %7, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.compare  EQ, %arg1, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = stablehlo.select %10, %cst, %arg1 : tensor<i1>, tensor<f32>
    %12 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %13 = stablehlo.select %12, %cst_2, %11 : tensor<i1>, tensor<f32>
    %14 = stablehlo.compare  EQ, %9, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %15 = stablehlo.select %14, %cst, %9 : tensor<i1>, tensor<f32>
    %16 = stablehlo.compare  NE, %9, %9,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.select %16, %cst_2, %15 : tensor<i1>, tensor<f32>
    %18 = stablehlo.compare  LT, %13, %17,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %19 = call @_where(%18, %arg2, %4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = call @_where(%18, %4, %arg3) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %19, %20 : tensor<i32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_0(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_1(%arg0: tensor<i1>, %arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @solve(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<256x256xf32>) -> (tensor<256x256xf32>, tensor<256xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<256xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<256x256xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<256x256xi1>, tensor<256x256xf32>
    %9 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<256xi32>) -> tensor<256xi32>
    %10 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %11 = call @_lu_solve(%8, %9, %10) : (tensor<256x256xf32>, tensor<256xi32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %12 : tensor<256x256xf32>
  }
  func.func private @_lu_solve(%arg0: tensor<256x256xf32>, %arg1: tensor<256xi32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x1xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<256xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<256xi1>, tensor<256xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 1, 1>}> : (tensor<256x256x1xf32>, tensor<256x1xi32>) -> tensor<256x256x1xf32>
    %8 = stablehlo.transpose %7, dims = [1, 2, 0] : (tensor<256x256x1xf32>) -> tensor<256x1x256xf32>
    %9 = stablehlo.reshape %8 : (tensor<256x1x256xf32>) -> tensor<256x256xf32>
    %10 = "stablehlo.triangular_solve"(%arg0, %9) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %11 = stablehlo.reshape %10 : (tensor<256x256xf32>) -> tensor<256x1x256xf32>
    %12 = stablehlo.reshape %11 : (tensor<256x1x256xf32>) -> tensor<256x256xf32>
    %13 = "stablehlo.triangular_solve"(%arg0, %12) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %14 = stablehlo.reshape %13 : (tensor<256x256xf32>) -> tensor<256x1x256xf32>
    %15 = stablehlo.slice %14 [0:256, 0:1, 0:256] : (tensor<256x1x256xf32>) -> tensor<256x1x256xf32>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1] : (tensor<256x1x256xf32>) -> tensor<256x256x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<256x256x1xf32>) -> tensor<256x256xf32>
    return %17 : tensor<256x256xf32>
  }
}
