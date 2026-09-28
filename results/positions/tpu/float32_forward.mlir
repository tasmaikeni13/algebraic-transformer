module @jit_ago_fwd attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x2048x8x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg1: tensor<64x2048x8x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg2: tensor<2048x32xf32>, %arg3: tensor<2048x32xf32>) -> (tensor<64x2048x8x64xf32> {jax.result_info = "result[0]"}, tensor<64x2048x8x64xf32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.reshape %arg2 : (tensor<2048x32xf32>) -> tensor<1x2048x1x32xf32>
    %1 = stablehlo.reshape %arg3 : (tensor<2048x32xf32>) -> tensor<1x2048x1x32xf32>
    %2 = stablehlo.reshape %arg0 : (tensor<64x2048x8x64xf32>) -> tensor<64x2048x8x32x2xf32>
    %3 = stablehlo.slice %2 [0:64, 0:2048, 0:8, 0:32, 0:1] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32xf32>
    %5 = stablehlo.slice %2 [0:64, 0:2048, 0:8, 0:32, 1:2] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32xf32>
    %7 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %8 = stablehlo.multiply %7, %4 : tensor<64x2048x8x32xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %10 = stablehlo.multiply %9, %6 : tensor<64x2048x8x32xf32>
    %11 = stablehlo.subtract %8, %10 : tensor<64x2048x8x32xf32>
    %12 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %13 = stablehlo.multiply %12, %4 : tensor<64x2048x8x32xf32>
    %14 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %15 = stablehlo.multiply %14, %6 : tensor<64x2048x8x32xf32>
    %16 = stablehlo.add %13, %15 : tensor<64x2048x8x32xf32>
    %17 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %19 = stablehlo.concatenate %17, %18, dim = 4 : (tensor<64x2048x8x32x1xf32>, tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32x2xf32>
    %20 = stablehlo.reshape %19 : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x64xf32>
    %21 = stablehlo.reshape %arg1 : (tensor<64x2048x8x64xf32>) -> tensor<64x2048x8x32x2xf32>
    %22 = stablehlo.slice %21 [0:64, 0:2048, 0:8, 0:32, 0:1] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32xf32>
    %24 = stablehlo.slice %21 [0:64, 0:2048, 0:8, 0:32, 1:2] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32xf32>
    %26 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %27 = stablehlo.multiply %26, %23 : tensor<64x2048x8x32xf32>
    %28 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %29 = stablehlo.multiply %28, %25 : tensor<64x2048x8x32xf32>
    %30 = stablehlo.subtract %27, %29 : tensor<64x2048x8x32xf32>
    %31 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %32 = stablehlo.multiply %31, %23 : tensor<64x2048x8x32xf32>
    %33 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %34 = stablehlo.multiply %33, %25 : tensor<64x2048x8x32xf32>
    %35 = stablehlo.add %32, %34 : tensor<64x2048x8x32xf32>
    %36 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %37 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %38 = stablehlo.concatenate %36, %37, dim = 4 : (tensor<64x2048x8x32x1xf32>, tensor<64x2048x8x32x1xf32>) -> tensor<64x2048x8x32x2xf32>
    %39 = stablehlo.reshape %38 : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x64xf32>
    return %20, %39 : tensor<64x2048x8x64xf32>, tensor<64x2048x8x64xf32>
  }
}
