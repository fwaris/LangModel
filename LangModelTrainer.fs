module LangModelTrainer
(*
Based on this tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
*)

open System
open TorchSharp
open type TorchSharp.torch.nn
open type TorchSharp.torch.optim
open type torch.TensorIndex
open TorchSharp.Fun
open System.Diagnostics
open TextUtils

//select model type by 'opening' the appropriate module below
open LangModelGpt        
//open LangModel

let bptt = 32L              //chunk size for pairing words together for training
let batch_size = 20L
let eval_batch_size = 64L
let epochs = 50
let logInterval = 200
let cmdArgs = Environment.GetCommandLineArgs()
let datasetPath = @"E:\s\langModel\wikitext-2"
let datasets = Wiki2.load datasetPath
let modelFile = Model.modelFile
let modelFn = Model.create
let modelType = Model.modelType

//index all tokens in the training set
let vocab = Vocabulary.create datasets.Train

torch.random.manual_seed(1L) |> ignore
let hasCUDA = torch.cuda_is_available()
let device = if hasCUDA then torch.CUDA else torch.CPU
let criterion x y = torch.nn.functional.cross_entropy(x,y,reduction=Reduction.Mean)

//convert lines of text to a sequence of token indexes and convert to a single tensor
let process_input (lines:string seq) = 
    torch.cat(
        [|
            for i in lines do
                let dt = [| for token in TextUtils.Tokenizer.tokenize(i) do
                            let tk = vocab |> Vocabulary.index token
                            yield tk
                         |]
                let t = torch.tensor(dt)
                if t.NumberOfElements > 0L then
                    t
        |],
        0L)

//ensure that the input tensor is evenly divisible by the select batch size
let batchify (data:torch.Tensor) batchSize (device:torch.Device) =
    let nbatch = data.shape.[0] / batchSize
    let d2 = data.narrow(0L,0L,nbatch*batchSize).view(batchSize, -1L).t()
    d2.contiguous().``to``(device)

//generate source and target batches given an index into the input tensor
let get_batch (input:torch.Tensor) (index:int64) =
    let len = min bptt (input.shape.[0]-1L-index)        //length of word sequence
    let words     = input.[Slice(index, index + len)]                          
    let nextWords = input.[Slice(index + 1L, index + 1L + len)]
    let target = nextWords.reshape(-1L) //flatten next words as loss is easier to calculate
    words,target
     
//
let train epoch (model:IModel) (optimizer:Optimizer) (trainData:torch.Tensor) ntokens =
    model.Module.train()
    let mutable total_loss = 0.0f
    let mutable src_mask =  TextUtils.Masks.generateSubsequentMask bptt device
    let mutable batch = 0
    let tdlen = trainData.shape.[0]
    let mutable i = 0L
    while i < tdlen - 2L  do
        begin
            let data,targets = get_batch trainData i
            use data = data
            use targets = targets
            if data.shape.[0] <> bptt then
                src_mask.Dispose()
                src_mask <-  TextUtils.Masks.generateSubsequentMask data.shape.[0] device
            optimizer.zero_grad()
            let args = Args()?mask<-src_mask
            let output,_ = model.forward(data, args)
            let outputView = output.view(-1L, ntokens)
            use loss = criterion outputView targets
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.Module.parameters(), 0.5) |> ignore
            optimizer.step() |> ignore
            total_loss <- total_loss + loss.cpu().item<float32>()
            output.Dispose()
        end 
        GC.Collect()
        if (batch % logInterval = 0) && (batch > 0) then
            let cur_loss = (total_loss / (float32 logInterval)).ToString("0.00")
            printfn $"epoch: {epoch} | batch: {batch} / {tdlen/bptt} | loss: {cur_loss}"
            total_loss <- 0.0f
        batch <- batch + 1
        i <- i + bptt


let evaluate (model:IModel) (evalData:torch.Tensor) ntokens vocabI epoch =
    model.Module.eval()
    let mutable total_loss = 0.0f
    let mutable src_mask =  TextUtils.Masks.generateSubsequentMask bptt device
    let mutable batch = 0L
    let tdlen = evalData.shape.[0]
    let mutable i = 0L
    while i < tdlen - 2L  do
        begin
            let data,targets = get_batch evalData i
            use data = data
            use targets = targets

            if data.shape.[0] <> bptt then
                src_mask.Dispose()
                src_mask <- TextUtils.Masks.generateSubsequentMask data.shape.[0] device
            let args = Args()?mask<-src_mask
            let output,_ = model.forward(data, args)
            use output = output
            //if epoch >= 5 then
            //    getWords data output |> ignore
            use loss = criterion (output.view(-1L, ntokens)) targets
            total_loss <- total_loss + (float32 data.shape.[0]) * loss.cpu().item<float32>()
        end 
        GC.Collect()
        batch <- batch + 1L
        i <- i + bptt
    total_loss / (float32 evalData.shape.[0])

let run epochs =
    printfn $"Running SequenceToSequence on {device.``type``.ToString()} for {epochs} epochs."
    let train_data = batchify (process_input datasets.Train) batch_size device
    let test_data = batchify (process_input datasets.Test) eval_batch_size device
    let valid_data = batchify (process_input datasets.Valid) eval_batch_size device
    let ntokens = Vocabulary.count vocab
    let imodel = modelFn ntokens device
    imodel.Module.``to``(device) |> ignore
    let lr = 2.50
    let optimizer = SGD(imodel.Module.parameters(), lr)
    let scheduler = lr_scheduler.StepLR(optimizer, 1, 0.95, last_epoch=15)
    let totalTime = Stopwatch()
    totalTime.Start()
    for epoch = 1 to epochs do
        let sw = Stopwatch()
        sw.Start()
        train epoch imodel optimizer train_data ntokens
        let val_loss = evaluate imodel valid_data ntokens vocab epoch
        sw.Stop()
        let elapsed = sw.Elapsed.TotalSeconds.ToString("0.0")
        let lossStr = val_loss.ToString("0.00")
        printfn $"\nEnd of epoch: {epoch} | time: {elapsed}s | loss: {lossStr}\n"
        scheduler.step()
        imodel.Module.save Model.modelFile |> ignore
        
    let tst_loss = evaluate imodel test_data ntokens vocab 1
    totalTime.Stop()
    let elapsed = totalTime.Elapsed.TotalSeconds.ToString("0.0")
    let lossStr = tst_loss.ToString("0.00")
    printfn $"\nEnd of training | time: {elapsed} s | loss: {lossStr}\n"


