module @jit_expm_before_fix attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256xf32>) -> (tensor<256x256xf32> {jax.result_info = ""}) {
    %0 = call @expm_before_fix(%arg0) : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @expm_before_fix(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<[0.0149558522, 0.253939837, 0.950417876, 2.09784794]> : tensor<4xf32>
    %0 = stablehlo.abs %arg0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst_1) applies stablehlo.maximum across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %3 = stablehlo.iota dim = 0 : tensor<256x256xi32>
    %4 = stablehlo.iota dim = 1 : tensor<256x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %6 = stablehlo.add %3, %5 : tensor<256x256xi32>
    %7 = stablehlo.compare  EQ, %6, %4,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %8 = stablehlo.convert %7 : (tensor<256x256xi1>) -> tensor<256x256xf32>
    %9 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<256x256xf32>
    %cst_3 = stablehlo.constant dense<6.000000e+01> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %13 = stablehlo.multiply %12, %8 : tensor<256x256xf32>
    %14 = stablehlo.add %11, %13 : tensor<256x256xf32>
    %15 = stablehlo.dot_general %arg0, %14, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_4 = stablehlo.constant dense<1.200000e+01> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %17 = stablehlo.multiply %16, %9 : tensor<256x256xf32>
    %cst_5 = stablehlo.constant dense<1.200000e+02> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %19 = stablehlo.multiply %18, %8 : tensor<256x256xf32>
    %20 = stablehlo.add %17, %19 : tensor<256x256xf32>
    %21 = stablehlo.iota dim = 0 : tensor<256x256xi32>
    %22 = stablehlo.iota dim = 1 : tensor<256x256xi32>
    %23 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %24 = stablehlo.add %21, %23 : tensor<256x256xi32>
    %25 = stablehlo.compare  EQ, %24, %22,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %26 = stablehlo.convert %25 : (tensor<256x256xi1>) -> tensor<256x256xf32>
    %27 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %28 = stablehlo.dot_general %27, %27, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %29 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %30 = stablehlo.multiply %29, %28 : tensor<256x256xf32>
    %cst_6 = stablehlo.constant dense<4.200000e+02> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %32 = stablehlo.multiply %31, %27 : tensor<256x256xf32>
    %33 = stablehlo.add %30, %32 : tensor<256x256xf32>
    %cst_7 = stablehlo.constant dense<1.512000e+04> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %35 = stablehlo.multiply %34, %26 : tensor<256x256xf32>
    %36 = stablehlo.add %33, %35 : tensor<256x256xf32>
    %37 = stablehlo.dot_general %arg0, %36, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_8 = stablehlo.constant dense<3.000000e+01> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %39 = stablehlo.multiply %38, %28 : tensor<256x256xf32>
    %cst_9 = stablehlo.constant dense<3.360000e+03> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %41 = stablehlo.multiply %40, %27 : tensor<256x256xf32>
    %42 = stablehlo.add %39, %41 : tensor<256x256xf32>
    %cst_10 = stablehlo.constant dense<3.024000e+04> : tensor<f32>
    %43 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %44 = stablehlo.multiply %43, %26 : tensor<256x256xf32>
    %45 = stablehlo.add %42, %44 : tensor<256x256xf32>
    %46 = stablehlo.iota dim = 0 : tensor<256x256xi32>
    %47 = stablehlo.iota dim = 1 : tensor<256x256xi32>
    %48 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %49 = stablehlo.add %46, %48 : tensor<256x256xi32>
    %50 = stablehlo.compare  EQ, %49, %47,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %51 = stablehlo.convert %50 : (tensor<256x256xi1>) -> tensor<256x256xf32>
    %52 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %53 = stablehlo.dot_general %52, %52, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %54 = stablehlo.dot_general %53, %52, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %55 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %56 = stablehlo.multiply %55, %54 : tensor<256x256xf32>
    %cst_11 = stablehlo.constant dense<1.512000e+03> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %58 = stablehlo.multiply %57, %53 : tensor<256x256xf32>
    %59 = stablehlo.add %56, %58 : tensor<256x256xf32>
    %cst_12 = stablehlo.constant dense<2.772000e+05> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %61 = stablehlo.multiply %60, %52 : tensor<256x256xf32>
    %62 = stablehlo.add %59, %61 : tensor<256x256xf32>
    %cst_13 = stablehlo.constant dense<8.648640e+06> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %64 = stablehlo.multiply %63, %51 : tensor<256x256xf32>
    %65 = stablehlo.add %62, %64 : tensor<256x256xf32>
    %66 = stablehlo.dot_general %arg0, %65, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_14 = stablehlo.constant dense<5.600000e+01> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %68 = stablehlo.multiply %67, %54 : tensor<256x256xf32>
    %cst_15 = stablehlo.constant dense<2.520000e+04> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %70 = stablehlo.multiply %69, %53 : tensor<256x256xf32>
    %71 = stablehlo.add %68, %70 : tensor<256x256xf32>
    %cst_16 = stablehlo.constant dense<1.995840e+06> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %73 = stablehlo.multiply %72, %52 : tensor<256x256xf32>
    %74 = stablehlo.add %71, %73 : tensor<256x256xf32>
    %cst_17 = stablehlo.constant dense<0x4B83F7C0> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %76 = stablehlo.multiply %75, %51 : tensor<256x256xf32>
    %77 = stablehlo.add %74, %76 : tensor<256x256xf32>
    %78 = stablehlo.iota dim = 0 : tensor<256x256xi32>
    %79 = stablehlo.iota dim = 1 : tensor<256x256xi32>
    %80 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %81 = stablehlo.add %78, %80 : tensor<256x256xi32>
    %82 = stablehlo.compare  EQ, %81, %79,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %83 = stablehlo.convert %82 : (tensor<256x256xi1>) -> tensor<256x256xf32>
    %84 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %85 = stablehlo.dot_general %84, %84, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %86 = stablehlo.dot_general %85, %84, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %87 = stablehlo.dot_general %86, %84, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %88 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %89 = stablehlo.multiply %88, %87 : tensor<256x256xf32>
    %cst_18 = stablehlo.constant dense<3.960000e+03> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %91 = stablehlo.multiply %90, %86 : tensor<256x256xf32>
    %92 = stablehlo.add %89, %91 : tensor<256x256xf32>
    %cst_19 = stablehlo.constant dense<2.162160e+06> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %94 = stablehlo.multiply %93, %85 : tensor<256x256xf32>
    %95 = stablehlo.add %92, %94 : tensor<256x256xf32>
    %cst_20 = stablehlo.constant dense<0x4D9056FA> : tensor<f32>
    %96 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %97 = stablehlo.multiply %96, %84 : tensor<256x256xf32>
    %98 = stablehlo.add %95, %97 : tensor<256x256xf32>
    %cst_21 = stablehlo.constant dense<8.82161254E+9> : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %100 = stablehlo.multiply %99, %83 : tensor<256x256xf32>
    %101 = stablehlo.add %98, %100 : tensor<256x256xf32>
    %102 = stablehlo.dot_general %arg0, %101, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_22 = stablehlo.constant dense<9.000000e+01> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %104 = stablehlo.multiply %103, %87 : tensor<256x256xf32>
    %cst_23 = stablehlo.constant dense<1.108800e+05> : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %106 = stablehlo.multiply %105, %86 : tensor<256x256xf32>
    %107 = stablehlo.add %104, %106 : tensor<256x256xf32>
    %cst_24 = stablehlo.constant dense<0x4BE6F190> : tensor<f32>
    %108 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %109 = stablehlo.multiply %108, %85 : tensor<256x256xf32>
    %110 = stablehlo.add %107, %109 : tensor<256x256xf32>
    %cst_25 = stablehlo.constant dense<2.0756736E+9> : tensor<f32>
    %111 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %112 = stablehlo.multiply %111, %84 : tensor<256x256xf32>
    %113 = stablehlo.add %110, %112 : tensor<256x256xf32>
    %cst_26 = stablehlo.constant dense<1.76432251E+10> : tensor<f32>
    %114 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %115 = stablehlo.multiply %114, %83 : tensor<256x256xf32>
    %116 = stablehlo.add %113, %115 : tensor<256x256xf32>
    %cst_27 = stablehlo.constant dense<5.37192059> : tensor<f32>
    %117 = stablehlo.divide %2, %cst_27 : tensor<f32>
    %118 = stablehlo.log %117 : tensor<f32>
    %cst_28 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %119 = stablehlo.log %cst_28 : tensor<f32>
    %120 = stablehlo.divide %118, %119 : tensor<f32>
    %121 = stablehlo.floor %120 : tensor<f32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %122 = stablehlo.maximum %cst_29, %121 : tensor<f32>
    %123 = stablehlo.power %cst_28, %122 : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %125 = stablehlo.divide %arg0, %124 : tensor<256x256xf32>
    %126 = stablehlo.iota dim = 0 : tensor<256x256xi32>
    %127 = stablehlo.iota dim = 1 : tensor<256x256xi32>
    %128 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %129 = stablehlo.add %126, %128 : tensor<256x256xi32>
    %130 = stablehlo.compare  EQ, %129, %127,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %131 = stablehlo.convert %130 : (tensor<256x256xi1>) -> tensor<256x256xf32>
    %132 = stablehlo.dot_general %125, %125, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %133 = stablehlo.dot_general %132, %132, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %134 = stablehlo.dot_general %133, %132, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %135 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %136 = stablehlo.multiply %135, %134 : tensor<256x256xf32>
    %cst_30 = stablehlo.constant dense<1.638000e+04> : tensor<f32>
    %137 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %138 = stablehlo.multiply %137, %133 : tensor<256x256xf32>
    %139 = stablehlo.add %136, %138 : tensor<256x256xf32>
    %cst_31 = stablehlo.constant dense<4.084080e+07> : tensor<f32>
    %140 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %141 = stablehlo.multiply %140, %132 : tensor<256x256xf32>
    %142 = stablehlo.add %139, %141 : tensor<256x256xf32>
    %143 = stablehlo.dot_general %134, %142, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_32 = stablehlo.constant dense<3.35221289E+10> : tensor<f32>
    %144 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %145 = stablehlo.multiply %144, %134 : tensor<256x256xf32>
    %146 = stablehlo.add %143, %145 : tensor<256x256xf32>
    %cst_33 = stablehlo.constant dense<1.05594707E+13> : tensor<f32>
    %147 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %148 = stablehlo.multiply %147, %133 : tensor<256x256xf32>
    %149 = stablehlo.add %146, %148 : tensor<256x256xf32>
    %cst_34 = stablehlo.constant dense<1.18735378E+15> : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %151 = stablehlo.multiply %150, %132 : tensor<256x256xf32>
    %152 = stablehlo.add %149, %151 : tensor<256x256xf32>
    %cst_35 = stablehlo.constant dense<3.23823762E+16> : tensor<f32>
    %153 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %154 = stablehlo.multiply %153, %131 : tensor<256x256xf32>
    %155 = stablehlo.add %152, %154 : tensor<256x256xf32>
    %156 = stablehlo.dot_general %125, %155, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_36 = stablehlo.constant dense<1.820000e+02> : tensor<f32>
    %157 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %158 = stablehlo.multiply %157, %134 : tensor<256x256xf32>
    %cst_37 = stablehlo.constant dense<9.609600e+05> : tensor<f32>
    %159 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %160 = stablehlo.multiply %159, %133 : tensor<256x256xf32>
    %161 = stablehlo.add %158, %160 : tensor<256x256xf32>
    %cst_38 = stablehlo.constant dense<1.32324198E+9> : tensor<f32>
    %162 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %163 = stablehlo.multiply %162, %132 : tensor<256x256xf32>
    %164 = stablehlo.add %161, %163 : tensor<256x256xf32>
    %165 = stablehlo.dot_general %134, %164, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %cst_39 = stablehlo.constant dense<6.70442586E+11> : tensor<f32>
    %166 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %167 = stablehlo.multiply %166, %134 : tensor<256x256xf32>
    %168 = stablehlo.add %165, %167 : tensor<256x256xf32>
    %cst_40 = stablehlo.constant dense<1.29060194E+14> : tensor<f32>
    %169 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %170 = stablehlo.multiply %169, %133 : tensor<256x256xf32>
    %171 = stablehlo.add %168, %170 : tensor<256x256xf32>
    %cst_41 = stablehlo.constant dense<7.771770e+15> : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %173 = stablehlo.multiply %172, %132 : tensor<256x256xf32>
    %174 = stablehlo.add %171, %173 : tensor<256x256xf32>
    %cst_42 = stablehlo.constant dense<6.47647525E+16> : tensor<f32>
    %175 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %176 = stablehlo.multiply %175, %131 : tensor<256x256xf32>
    %177 = stablehlo.add %174, %176 : tensor<256x256xf32>
    %178 = stablehlo.slice %cst [0:1] : (tensor<4xf32>) -> tensor<1xf32>
    %179 = stablehlo.reshape %178 : (tensor<1xf32>) -> tensor<f32>
    %180 = stablehlo.compare  LT, %2, %179,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %181 = stablehlo.slice %cst [1:2] : (tensor<4xf32>) -> tensor<1xf32>
    %182 = stablehlo.reshape %181 : (tensor<1xf32>) -> tensor<f32>
    %183 = stablehlo.compare  LT, %2, %182,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %184 = stablehlo.slice %cst [2:3] : (tensor<4xf32>) -> tensor<1xf32>
    %185 = stablehlo.reshape %184 : (tensor<1xf32>) -> tensor<f32>
    %186 = stablehlo.compare  LT, %2, %185,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %187 = stablehlo.slice %cst [3:4] : (tensor<4xf32>) -> tensor<1xf32>
    %188 = stablehlo.reshape %187 : (tensor<1xf32>) -> tensor<f32>
    %189 = stablehlo.compare  LT, %2, %188,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_43 = stablehlo.constant dense<false> : tensor<i1>
    %190 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %191 = stablehlo.broadcast_in_dim %180, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %192 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %193 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %194 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %195 = stablehlo.concatenate %190, %191, %192, %193, %194, dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<5xi1>
    %196 = call @argmax(%195) : (tensor<5xi1>) -> tensor<i32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %c_44 = stablehlo.constant dense<2> : tensor<i32>
    %198 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %199 = stablehlo.compare  LT, %197, %198,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %c_45 = stablehlo.constant dense<1> : tensor<i32>
    %200 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %201 = stablehlo.compare  LT, %197, %200,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %202 = stablehlo.select %201, %156, %15 : tensor<256x256xi1>, tensor<256x256xf32>
    %c_46 = stablehlo.constant dense<3> : tensor<i32>
    %203 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %204 = stablehlo.compare  LT, %197, %203,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %c_47 = stablehlo.constant dense<4> : tensor<i32>
    %205 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %206 = stablehlo.compare  LT, %197, %205,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %207 = stablehlo.select %206, %66, %102 : tensor<256x256xi1>, tensor<256x256xf32>
    %208 = stablehlo.select %204, %37, %207 : tensor<256x256xi1>, tensor<256x256xf32>
    %209 = stablehlo.select %199, %202, %208 : tensor<256x256xi1>, tensor<256x256xf32>
    %210 = stablehlo.slice %cst [0:1] : (tensor<4xf32>) -> tensor<1xf32>
    %211 = stablehlo.reshape %210 : (tensor<1xf32>) -> tensor<f32>
    %212 = stablehlo.compare  LT, %2, %211,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %213 = stablehlo.slice %cst [1:2] : (tensor<4xf32>) -> tensor<1xf32>
    %214 = stablehlo.reshape %213 : (tensor<1xf32>) -> tensor<f32>
    %215 = stablehlo.compare  LT, %2, %214,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %216 = stablehlo.slice %cst [2:3] : (tensor<4xf32>) -> tensor<1xf32>
    %217 = stablehlo.reshape %216 : (tensor<1xf32>) -> tensor<f32>
    %218 = stablehlo.compare  LT, %2, %217,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %219 = stablehlo.slice %cst [3:4] : (tensor<4xf32>) -> tensor<1xf32>
    %220 = stablehlo.reshape %219 : (tensor<1xf32>) -> tensor<f32>
    %221 = stablehlo.compare  LT, %2, %220,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %222 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %223 = stablehlo.broadcast_in_dim %212, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %224 = stablehlo.broadcast_in_dim %215, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %225 = stablehlo.broadcast_in_dim %218, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %226 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %227 = stablehlo.concatenate %222, %223, %224, %225, %226, dim = 0 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<5xi1>
    %228 = call @argmax(%227) : (tensor<5xi1>) -> tensor<i32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %c_48 = stablehlo.constant dense<2> : tensor<i32>
    %230 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %231 = stablehlo.compare  LT, %229, %230,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %c_49 = stablehlo.constant dense<1> : tensor<i32>
    %232 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %233 = stablehlo.compare  LT, %229, %232,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %234 = stablehlo.select %233, %177, %20 : tensor<256x256xi1>, tensor<256x256xf32>
    %c_50 = stablehlo.constant dense<3> : tensor<i32>
    %235 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %236 = stablehlo.compare  LT, %229, %235,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %c_51 = stablehlo.constant dense<4> : tensor<i32>
    %237 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i32>) -> tensor<256x256xi32>
    %238 = stablehlo.compare  LT, %229, %237,  SIGNED : (tensor<256x256xi32>, tensor<256x256xi32>) -> tensor<256x256xi1>
    %239 = stablehlo.select %238, %77, %116 : tensor<256x256xi1>, tensor<256x256xf32>
    %240 = stablehlo.select %236, %45, %239 : tensor<256x256xi1>, tensor<256x256xf32>
    %241 = stablehlo.select %231, %234, %240 : tensor<256x256xi1>, tensor<256x256xf32>
    %242 = stablehlo.add %209, %241 : tensor<256x256xf32>
    %243 = stablehlo.negate %209 : tensor<256x256xf32>
    %244 = stablehlo.add %243, %241 : tensor<256x256xf32>
    %245 = call @solve(%244, %242) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %246 = stablehlo.convert %122 : (tensor<f32>) -> tensor<i32>
    %c_52 = stablehlo.constant dense<0> : tensor<i32>
    %247:3 = stablehlo.while(%iterArg = %c_52, %iterArg_53 = %246, %iterArg_54 = %245) : tensor<i32>, tensor<i32>, tensor<256x256xf32>
     cond {
      %248 = stablehlo.compare  LT, %iterArg, %iterArg_53,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %248 : tensor<i1>
    } do {
      %c_55 = stablehlo.constant dense<1> : tensor<i32>
      %248 = stablehlo.add %iterArg, %c_55 : tensor<i32>
      %249 = stablehlo.dot_general %iterArg_54, %iterArg_54, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      stablehlo.return %248, %iterArg_53, %249 : tensor<i32>, tensor<i32>, tensor<256x256xf32>
    }
    return %247#2 : tensor<256x256xf32>
  }
  func.func private @argmax(%arg0: tensor<5xi1>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<5xi32>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %c), (%0 init: %c_0) across dimensions = [0] : (tensor<5xi1>, tensor<5xi32>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i32>)
     reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i1>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<i1>, tensor<i32>
    }
    return %1#1 : tensor<i32>
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
