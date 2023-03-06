module LangModelGen
open System
open TorchSharp
open TorchSharp.Fun
open TextUtils

let rng = System.Random()

let loadModel ntokens =
    let m = LangModelTrainer.modelFn ntokens torch.CPU
    m.Module.load(LangModelTrainer.modelFile).to'(LangModelTrainer.device) |> ignore
    m

let nextWord (input:torch.Tensor) (output:torch.Tensor) =
    let temperature = 0.001f // controls the range when sampling for the next word
    let nextWordScores = output.[-1].unsqueeze(0) //scores for the next word - following the last word in the input 
    let rawScores = 
        Tensor.getData<float32> nextWordScores //next word scores as float values
        |> Seq.indexed                         //tag each position with index
        |> Seq.skip (TextUtils.Vocabulary.spSet.Count) //skip specials for output
        |> Seq.toArray 
    let posScores = rawScores |> Array.filter (fun (_,x) -> x > 0.f)   //consider only positive scores
    let sumScores = posScores |> Array.sumBy snd                       //sum of scores for probability calculation
    let posProb = posScores |> Array.map (fun (i,x) -> i,x/sumScores)  //scores converted to probabilities
    let cumsum = 
        (posProb |> Array.sortByDescending snd,(0,0.f))                //sort by score
        ||> Array.scanBack(fun (j,x) (_,acc) -> j,acc+x)               //cumulative sum of probabilities needed for sampling
    let rand = 1.0f - (rng.NextSingle() * temperature)                 //sampling value
    let nextWordIdx = cumsum |> Seq.pick (fun (i,x) -> if x < rand then Some i else None) //selected sample word index
    let nextWord = Vocabulary.value nextWordIdx LangModelTrainer.vocab                    //actual word at index
    nextWord

let generate device vocab (model:IModel) (prompt:string list) =

    model.Module.eval()

    //generation loop - select a next word add it to input sequence and continue till max words reached
    let rec loop c maxC (ws:string list) =
        if c >= maxC then
            ws
        else
            //printfn "%A" ws
            let data = LangModelTrainer.process_input ws
            let dbatch = LangModelTrainer.batchify data 1L device
            let src_mask = TextUtils.Masks.generateSubsequentMask data.shape.[0] device
            let args = Args()?mask<-src_mask
            let output,_ = model.forward(dbatch,args)
            let nextWord = nextWord dbatch output
            loop (c+1) maxC (ws @ [nextWord])

    let words = loop 0 32 prompt
    let str = String.Join(" ", words)
    str