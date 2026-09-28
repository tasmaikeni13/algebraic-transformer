module @jit_oace_fwd attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x1024xf32> {mhlo.sharding = "{devices=[16,1]<=[16]}"}, %arg1: tensor<256xi32> {mhlo.sharding = "{devices=[16]<=[16]}"}) -> (tensor<256xf32> {jax.result_info = "result"}) {
    %0 = call @oace_loss(%arg0, %arg1) : (tensor<256x1024xf32>, tensor<256xi32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @oace_loss(%arg0: tensor<256x1024xf32>, %arg1: tensor<256xi32>) -> tensor<256xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<256x1024xi32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1024xi32>
    %5 = stablehlo.compare  EQ, %3, %4,  SIGNED : (tensor<256x1024xi32>, tensor<256x1024xi32>) -> tensor<256x1024xi1>
    %6 = stablehlo.convert %5 : (tensor<256x1024xi1>) -> tensor<256x1024xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %8 = stablehlo.rsqrt %arg0 : tensor<256x1024xf32>
    %9 = stablehlo.rsqrt %8 : tensor<256x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x1024xf32>
    %11 = stablehlo.multiply %6, %10 : tensor<256x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_1 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %14 = stablehlo.multiply %13, %12 : tensor<256xf32>
    %15 = stablehlo.multiply %arg0, %10 : tensor<256x1024xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_3 = stablehlo.constant dense<1.14285719> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %18 = stablehlo.multiply %17, %16 : tensor<256xf32>
    %19 = stablehlo.add %14, %18 : tensor<256xf32>
    %cst_4 = stablehlo.constant dense<9.14285755> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %21 = stablehlo.multiply %20, %7 : tensor<256xf32>
    %22 = stablehlo.subtract %19, %21 : tensor<256xf32>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %24 = stablehlo.multiply %23, %22 : tensor<256xf32>
    return %24 : tensor<256xf32>
  }
}
