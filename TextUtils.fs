module TextUtils 
(*
Some code based on TorchSharp Example Utilities
https://github.com/dotnet/TorchSharp/tree/f8e8ad98d38be0e612ec031380af3cafc51f2cbf/src/Examples.Utils
*)

module Tokenizer =
    open System.Text.RegularExpressions
    let ptrnRplts =
        [|
            "\'"      ," \\'  "
            "\""       , ""
            "\\."      , " . "
            "<br \\/>" ," "
            ","        ," , "
            "\\("      , " ( "
            "\\)"      ," ) "
            "\\!"      ," ! "
            "\\?"      ," ? "
            "\\;"      , " "
            "\\:"      , " "
            "\\\\"     , " "
            "\\s+"     ," "
        |]
    let tokenize (input:string) =
        let input' = 
            (input,ptrnRplts) 
            ||> Array.fold (fun acc (p,r) -> Regex.Replace(acc,p,r))
        input'.Split(' ')
        
module Vocabulary =
    open System.Collections.Generic

    type Vocab = {Text2I:IDictionary<string,int64>; I2Text:string[]} //2-way index

    let unk = "<unk>"
    
    let specials = [unk]
    let spSet = set specials
  
    let create text =
        let tokens =
            text
            |> Seq.collect Tokenizer.tokenize
            |> Seq.distinct
            |> Seq.filter (spSet.Contains>>not)
            |> Seq.append specials
            |> Seq.toArray
        let textToI = 
            tokens
            |> Seq.indexed
            |> Seq.map (fun (i,x) -> x,int64 i)
            |> dict
        {Text2I = textToI; I2Text = tokens}

    let index (s:string) (vocab:Vocab)  =
        match vocab.Text2I.TryGetValue s with
        | true,i -> i
        | _      -> vocab.Text2I.[unk]

    let value (i:int64) vocab = vocab.I2Text.[int i]

    let count vocab = vocab.I2Text.Length

module Masks =
    open TorchSharp
    open System
    let generateSubsequentMask size (device:TorchSharp.torch.Device) =
        let mask = torch.ones([|size; size|]).tril()
        let subseqMask=torch.zeros(size,size).masked_fill(mask.eq(0.f.ToScalar()), (-infinityf).ToScalar())
        subseqMask.``to``(device)