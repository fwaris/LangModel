module LangModel_Pgm
open System
let generate() =
    let device = LangModelTrainer.device
    let vocab = LangModelTrainer.vocab
    let ntokens = TextUtils.Vocabulary.count vocab
    let model = LangModelGen.loadModel ntokens
    printfn "model loaded"

    let replacements = ["@-@","-"; "@,@", ","]

    let rec generate() =
        printfn "enter a few words to prompt text generation OR q to quit"
        let line = Console.ReadLine()
        match line.Trim() with 
        | "" -> generate()
        | "q" -> ()
        | words -> 
            let prompt = words.Split(" ",StringSplitOptions.RemoveEmptyEntries) |> Array.toList
            printfn "====>"
            let str = LangModelGen.generate device vocab model prompt
            let str = (str,replacements) ||> List.fold(fun acc (f,r) -> acc.Replace(f,r))
            printfn "%s" str
            printfn "-------------"
            generate()

    generate()


[<EntryPoint>]
let main args =
    printfn $"Model type: {LangModelTrainer.modelType}"
    printfn ""
    let rec loop (args:string[]) = 
        let args = args |> Array.toList |> List.map (fun x -> x.ToLower())
        match args with
        | "train":: _ -> 
            LangModelTrainer.run 20  // train the mo
            printfn "done training"
        | "gen"::_ -> generate()
        | _ -> 
            printfn "Enter: train - to train model OR gen - to generate sentences from a trained model"
            let line = Console.ReadLine()
            if String.IsNullOrWhiteSpace line then
                ()
            else
                loop (line.Trim().Split(" ",StringSplitOptions.RemoveEmptyEntries))
    loop args
    0