module LangModel
(*
#r "nuget: torchsharp-cuda-windows"
let dmodel = 200
*)
open TorchSharp
open System
open TorchSharp.Fun

(*
Model based on this tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
encoder 
*)

let emsize = 200L
let nhidden = 200L
let nlayers = 3L
let nheads = 20L
let dropout = 0.2

let IDX_NULL = Nullable<int64>()
let IDX_ZERO = torch.TensorIndex.Slice(0L)

//https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
module PositionalEncoder =
    open type torch.TensorIndex
    let private ``...`` = Ellipsis
    let private ``:`` = Colon
    let create dmodel maxLen =
        let dropout = torch.nn.Dropout(dropout) |> M
        let posTnsr = torch.zeros([|maxLen; dmodel|])       //5000 x 200
        let position = torch.arange(maxLen).unsqueeze(1L)   //5000 x 1
        let divTerm1 = -(10000.0.ToTensor().log()) / ((float dmodel).ToScalar())
        let divTerm2 = torch.arange(0L,dmodel,2L)               //100
        let divTerm3 = (divTerm2 * divTerm1.ToScalar())
        let divTerm = divTerm3.exp()   //100
        let divTermT = Tensor.getDataNested<float32> divTerm
        posTnsr.[ ``:``, Slice(0L,IDX_NULL,2L) ] <- (position * divTerm).sin()
        posTnsr.[ ``:``, Slice(1L,IDX_NULL,2L) ] <- (position * divTerm).cos()
        let pe = posTnsr.unsqueeze(0L).transpose(0L,1L)
        pe.name <- "pe"
        let peRef = ref pe

        let mdl = 
            F [] [dropout; peRef] (fun t -> 
                let pos = peRef.Value[Slice(IDX_NULL,t.shape.[0]), Slice()]
                use x = t + pos
                dropout.forward(x))
        mdl

module Model =
    let create ntokens (device:TorchSharp.torch.Device) =
        let pos_encoder = PositionalEncoder.create emsize 5000L
        let encoder_layer = torch.nn.TransformerEncoderLayer(emsize,nheads,nhidden,dropout)
        let transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,nlayers)
        let encoder = torch.nn.Embedding(ntokens,emsize)
        let decoder = torch.nn.Linear(emsize,ntokens)
        let sqrtEmbSz = (sqrt (float emsize)).ToScalar()

        let initRange = 0.1
        torch.nn.init.uniform_(encoder.weight,-initRange,initRange)   |> ignore
        torch.nn.init.zeros_(decoder.bias)                            |> ignore
        torch.nn.init.uniform_(decoder.weight, -initRange, initRange) |> ignore

        let mdl = 
            Fx [] [pos_encoder; transformer_encoder; encoder; decoder]  (fun (t,args) -> 
                let mask : torch.Tensor = args?mask
                let src = pos_encoder.forward(encoder.forward(t) * sqrtEmbSz)
                let enc = transformer_encoder.forward(src,mask)
                let dec = decoder.forward(enc)
                dec,args
            )

        mdl.to'(device)
        mdl

    let modelFile = @"E:\s\langModel\lang_model_enc.bin"

    let modelType = "Pytorch tutorial"
