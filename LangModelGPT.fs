module LangModelGpt

(*
GPT inspired decoder-only model with positional *embeddings* (as opposed to positional *encoding* )

Model structure is intented to match GPT architecture. It uses the built-in PyTorch TransformerEncoder module
and therefore looks different from some other GPT implementations
*)
open TorchSharp
open System
open TorchSharp.Fun


let emsize = 200L
let nhidden = 200L
let nlayers = 3L
let nheads = 10L
let dropout = 0.2
let maxSeq = 128L

let IDX_NULL = Nullable<int64>()
let IDX_ZERO = torch.TensorIndex.Slice(0L)


module Model =

    let create ntokens (device:TorchSharp.torch.Device) =
        let pos_emb = torch.nn.Embedding(maxSeq,emsize)
        let tok_emb = torch.nn.Embedding(ntokens,emsize)
        let txl = torch.nn.TransformerEncoderLayer(emsize,nheads,nhidden,dropout,activation=torch.nn.Activations.GELU)
        let tx = torch.nn.TransformerEncoder(txl,nlayers)
        let decoder = torch.nn.Linear(emsize,ntokens,hasBias=false)
        let inputDrop = torch.nn.Dropout(0.1)        

        let initRange = 0.1
        torch.nn.init.uniform_(tok_emb.weight,-initRange,initRange)   |> ignore
        torch.nn.init.uniform_(decoder.weight, -initRange, initRange) |> ignore

        let mdl = 
            Fx [] [pos_emb; tx; tok_emb; decoder; inputDrop;]  (fun (t,args) -> 
                let seqLen,batchSz = let s = t.size() in s.[0], s.[1]
                let mask : torch.Tensor = args?mask // s . s
                use pos = torch.arange(0,seqLen,dtype=torch.long, device=t.device)
                use pEmb = pos_emb.forward(pos)
                use pEmb1 = pEmb.unsqueeze(0).broadcast_to(batchSz,seqLen,emsize)
                use pEmb2 = pEmb1.permute(1,0,2)
                use tEmb = tok_emb.forward(t)
                use emb = tEmb + pEmb2
                use drp = inputDrop.forward(emb)
                use enc = tx.forward(drp,mask)
                let dec = decoder.forward(enc)
                dec,args
            )

        mdl.to'(device)
        mdl

    let modelFile = @"E:\s\langModel\lang_model_dec.bin"

    let modelType = "GPT arch."
