module @jit_ago_both attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x2048x8x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg1: tensor<64x2048x8x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg2: tensor<64x2048x8x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg3: tensor<2048x32xf32>, %arg4: tensor<2048x32xf32>) -> (tensor<64x2048x8x64xf32> {jax.result_info = "result[0][0]"}, tensor<64x2048x8x64xf32> {jax.result_info = "result[0][1]"}, tensor<64x2048x8x64xf32> {jax.result_info = "result[1][0]"}, tensor<64x2048x8x64xf32> {jax.result_info = "result[1][1]"}) {
    %0 = stablehlo.reshape %arg3 : (tensor<2048x32xf32>) -> tensor<1x2048x1x32xf32>
    %1 = stablehlo.reshape %arg4 : (tensor<2048x32xf32>) -> tensor<1x2048x1x32xf32>
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
    %40 = stablehlo.reshape %arg2 : (tensor<64x2048x8x64xf32>) -> tensor<64x2048x8x32x2xf32>
    %41 = stablehlo.slice %40 [0:64, 0:2048, 0:8, 0:32, 0:1] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %42 = stablehlo.slice %40 [0:64, 0:2048, 0:8, 0:32, 1:2] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%42 init: %cst) applies stablehlo.add across dimensions = [4] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%41 init: %cst_0) applies stablehlo.add across dimensions = [4] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32xf32>
    %45 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %46 = stablehlo.multiply %45, %43 : tensor<64x2048x8x32xf32>
    %47 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %48 = stablehlo.multiply %47, %43 : tensor<64x2048x8x32xf32>
    %49 = stablehlo.negate %44 : tensor<64x2048x8x32xf32>
    %50 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %51 = stablehlo.multiply %50, %49 : tensor<64x2048x8x32xf32>
    %52 = stablehlo.add %46, %51 : tensor<64x2048x8x32xf32>
    %53 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %54 = stablehlo.multiply %53, %44 : tensor<64x2048x8x32xf32>
    %55 = stablehlo.add %48, %54 : tensor<64x2048x8x32xf32>
    %56 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %57 = stablehlo.pad %56, %cst_1, low = [0, 0, 0, 0, 1], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32x2xf32>
    %58 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %59 = stablehlo.pad %58, %cst_1, low = [0, 0, 0, 0, 0], high = [0, 0, 0, 0, 1], interior = [0, 0, 0, 0, 0] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32x2xf32>
    %60 = stablehlo.add %57, %59 : tensor<64x2048x8x32x2xf32>
    %61 = stablehlo.reshape %60 : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x64xf32>
    %62 = stablehlo.reshape %arg2 : (tensor<64x2048x8x64xf32>) -> tensor<64x2048x8x32x2xf32>
    %63 = stablehlo.slice %62 [0:64, 0:2048, 0:8, 0:32, 0:1] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %64 = stablehlo.slice %62 [0:64, 0:2048, 0:8, 0:32, 1:2] : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x32x1xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %65 = stablehlo.reduce(%64 init: %cst_2) applies stablehlo.add across dimensions = [4] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = stablehlo.reduce(%63 init: %cst_3) applies stablehlo.add across dimensions = [4] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32xf32>
    %67 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %68 = stablehlo.multiply %67, %65 : tensor<64x2048x8x32xf32>
    %69 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %70 = stablehlo.multiply %69, %65 : tensor<64x2048x8x32xf32>
    %71 = stablehlo.negate %66 : tensor<64x2048x8x32xf32>
    %72 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %73 = stablehlo.multiply %72, %71 : tensor<64x2048x8x32xf32>
    %74 = stablehlo.add %68, %73 : tensor<64x2048x8x32xf32>
    %75 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2048x1x32xf32>) -> tensor<64x2048x8x32xf32>
    %76 = stablehlo.multiply %75, %66 : tensor<64x2048x8x32xf32>
    %77 = stablehlo.add %70, %76 : tensor<64x2048x8x32xf32>
    %78 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %79 = stablehlo.pad %78, %cst_1, low = [0, 0, 0, 0, 1], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32x2xf32>
    %80 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<64x2048x8x32xf32>) -> tensor<64x2048x8x32x1xf32>
    %81 = stablehlo.pad %80, %cst_1, low = [0, 0, 0, 0, 0], high = [0, 0, 0, 0, 1], interior = [0, 0, 0, 0, 0] : (tensor<64x2048x8x32x1xf32>, tensor<f32>) -> tensor<64x2048x8x32x2xf32>
    %82 = stablehlo.add %79, %81 : tensor<64x2048x8x32x2xf32>
    %83 = stablehlo.reshape %82 : (tensor<64x2048x8x32x2xf32>) -> tensor<64x2048x8x64xf32>
    return %20, %39, %83, %61 : tensor<64x2048x8x64xf32>, tensor<64x2048x8x64xf32>, tensor<64x2048x8x64xf32>, tensor<64x2048x8x64xf32>
  }
}
