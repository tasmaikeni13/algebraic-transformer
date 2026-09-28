module @jit_oace_fwd attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x1024xbf16> {mhlo.sharding = "{devices=[16,1]<=[16]}"}, %arg1: tensor<256xi32> {mhlo.sharding = "{devices=[16]<=[16]}"}) -> (tensor<256xbf16> {jax.result_info = "result"}) {
    %0 = call @oace_loss(%arg0, %arg1) : (tensor<256x1024xbf16>, tensor<256xi32>) -> tensor<256xbf16>
    return %0 : tensor<256xbf16>
  }
  func.func private @oace_loss(%arg0: tensor<256x1024xbf16>, %arg1: tensor<256xi32>) -> tensor<256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xbf16>) -> tensor<256x1024xf32>
    %1 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<256x1024xi32>
    %5 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1024xi32>
    %6 = stablehlo.compare  EQ, %4, %5,  SIGNED : (tensor<256x1024xi32>, tensor<256x1024xi32>) -> tensor<256x1024xi1>
    %7 = stablehlo.convert %6 : (tensor<256x1024xi1>) -> tensor<256x1024xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %9 = stablehlo.rsqrt %0 : tensor<256x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x1024xf32>
    %11 = stablehlo.rsqrt %10 : tensor<256x1024xf32>
    %12 = stablehlo.multiply %7, %11 : tensor<256x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_1 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %15 = stablehlo.multiply %14, %13 : tensor<256xf32>
    %16 = stablehlo.multiply %0, %11 : tensor<256x1024xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_3 = stablehlo.constant dense<1.14285719> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %19 = stablehlo.multiply %18, %17 : tensor<256xf32>
    %20 = stablehlo.add %15, %19 : tensor<256xf32>
    %cst_4 = stablehlo.constant dense<9.14285755> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %22 = stablehlo.multiply %21, %8 : tensor<256xf32>
    %23 = stablehlo.subtract %20, %22 : tensor<256xf32>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %25 = stablehlo.multiply %24, %23 : tensor<256xf32>
    %26 = stablehlo.convert %25 : (tensor<256xf32>) -> tensor<256xbf16>
    return %26 : tensor<256xbf16>
  }
}
