module @jit_algebraic_softmax attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<131072x128xf32> {mhlo.sharding = "{devices=[16,1]<=[16]}"}) -> (tensor<131072x128xf32> {jax.result_info = "result"}) {
    %0 = call @algebraic_softmax(%arg0) : (tensor<131072x128xf32>) -> tensor<131072x128xf32>
    return %0 : tensor<131072x128xf32>
  }
  func.func private @algebraic_softmax(%arg0: tensor<131072x128xf32>) -> tensor<131072x128xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<131072x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<131072x128xf32>, tensor<f32>) -> tensor<131072xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<131072xf32>) -> tensor<131072x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<131072x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<131072x1xf32>
    %7 = stablehlo.rsqrt %6 : tensor<131072x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<131072x1xf32>) -> tensor<131072x128xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<131072x128xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<131072x128xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %12 = stablehlo.add %11, %10 : tensor<131072x128xf32>
    %13 = stablehlo.rsqrt %12 : tensor<131072x128xf32>
    %14 = stablehlo.multiply %9, %13 : tensor<131072x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %16 = stablehlo.compare  LT, %9, %15,  FLOAT : (tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xi1>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %18 = stablehlo.subtract %17, %14 : tensor<131072x128xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %19 = call @_where(%16, %18, %cst_4) : (tensor<131072x128xi1>, tensor<131072x128xf32>, tensor<f32>) -> tensor<131072x128xf32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %21 = stablehlo.compare  LT, %9, %20,  FLOAT : (tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xi1>
    %22 = stablehlo.divide %13, %19 : tensor<131072x128xf32>
    %23 = stablehlo.multiply %9, %9 : tensor<131072x128xf32>
    %24 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<131072x128xf32>
    %25 = stablehlo.add %24, %23 : tensor<131072x128xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<131072x128xf32>
    %27 = stablehlo.add %9, %26 : tensor<131072x128xf32>
    %28 = call @_where_0(%21, %22, %27) : (tensor<131072x128xi1>, tensor<131072x128xf32>, tensor<131072x128xf32>) -> tensor<131072x128xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<131072x128xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<131072x128xf32>
    %31 = stablehlo.multiply %30, %30 : tensor<131072x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%31 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<131072x128xf32>, tensor<f32>) -> tensor<131072xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<131072xf32>) -> tensor<131072x1xf32>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<131072x1xf32>
    %35 = stablehlo.add %33, %34 : tensor<131072x1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1] : (tensor<131072x1xf32>) -> tensor<131072x128xf32>
    %37 = stablehlo.divide %31, %36 : tensor<131072x128xf32>
    return %37 : tensor<131072x128xf32>
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
