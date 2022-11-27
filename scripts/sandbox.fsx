#load "packages.fsx"
open TorchSharp
open TorchSharp.Fun

let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
let m1 = TextUtils.Masks.generateSubsequentMask 5 device

let m1T = Tensor.getDataNested<float32> m1

let memb = torch.nn.Embedding(5,10)
let p1 = torch.arange(0,5)
let p1T = Tensor.getDataNested<int64> p1
let p3 = p1.unsqueeze(0).broadcast_to(10,5)
let p3T = Tensor.getDataNested<int64> p3

let e2 = memb.forward p1
let e2T = Tensor.getDataNested<float32> e2
let e3 = e2.unsqueeze(0).broadcast_to(2,5,10)
let e3T = Tensor.getDataNested<float32> e3

