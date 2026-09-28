module @jit_algebraic_softmax attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<131072x128xbf16> {mhlo.sharding = "{devices=[16,1]<=[16]}"}) -> (tensor<131072x128xbf16> {jax.result_info = "result"}) {
    %0 = call @algebraic_softmax(%arg0) : (tensor<131072x128xbf16>) -> tensor<131072x128xbf16>
    return %0 : tensor<131072x128xbf16>
  }
  func.func private @algebraic_softmax(%arg0: tensor<131072x128xbf16>) -> tensor<131072x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<131072x128xbf16>) -> tensor<131072x128xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<131072x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<131072x128xf32>, tensor<f32>) -> tensor<131072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<131072xf32>) -> tensor<131072x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<131072x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<131072x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<131072x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<131072x1xf32>) -> tensor<131072x128xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<131072x128xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<131072x128xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %13 = stablehlo.add %12, %11 : tensor<131072x128xf32>
    %14 = stablehlo.rsqrt %13 : tensor<131072x128xf32>
    %15 = stablehlo.multiply %10, %14 : tensor<131072x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %17 = stablehlo.compare  LT, %10, %16,  FLOAT : (tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xi1>
    %18 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %19 = stablehlo.subtract %18, %15 : tensor<131072x128xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = call @_where(%17, %19, %cst_4) : (tensor<131072x128xi1>, tensor<131072x128xf32>, tensor<f32>) -> tensor<131072x128xf32>
    %21 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %22 = stablehlo.compare  LT, %10, %21,  FLOAT : (tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xi1>
    %23 = stablehlo.divide %14, %20 : tensor<131072x128xf32>
    %24 = stablehlo.multiply %10, %10 : tensor<131072x128xf32>
    %25 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %26 = stablehlo.add %25, %24 : tensor<131072x128xf32>
    %27 = stablehlo.multiply %26, %14 : tensor<131072x128xf32>
    %28 = stablehlo.add %10, %27 : tensor<131072x128xf32>
    %29 = call @_where_0(%22, %23, %28) : (tensor<131072x128xi1>, tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<131072x128xf32>
    %31 = stablehlo.multiply %30, %30 : tensor<131072x128xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<131072x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.reduce(%32 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<131072x128xf32>, tensor<f32>) -> tensor<131072xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0] : (tensor<131072xf32>) -> tensor<131072x1xf32>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %36 = stablehlo.add %34, %35 : tensor<131072x1xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<131072x1xf32>) -> tensor<131072x128xf32>
    %38 = stablehlo.divide %32, %37 : tensor<131072x128xf32>
    %39 = stablehlo.convert %38 : (tensor<131072x128xf32>) -> tensor<131072x128xbf16>
    return %39 : tensor<131072x128xbf16>
  }
  func.func private @_where(%arg0: tensor<131072x128xi1>, %arg1: tensor<131072x128xf32>, %arg2: tensor<f32>) -> tensor<131072x128xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<131072x128xi1>, tensor<131072x128xf32>
    return %2 : tensor<131072x128xf32>
  }
  func.func private @_where_0(%arg0: tensor<131072x128xi1>, %arg1: tensor<131072x128xf32>, %arg2: tensor<131072x128xf32>) -> tensor<131072x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<131072x128xi1>, tensor<131072x128xf32>
    return %0 : tensor<131072x128xf32>
  }
}
