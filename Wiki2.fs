module Wiki2
open System.IO

// data set download link
// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
let load root =
    let (@@) a b = Path.Combine(a,b)
    let test = root @@ "wiki.test.tokens"
    let train = root @@ "wiki.train.tokens"
    let valid = root @@ "wiki.train.tokens"
    let readFile (f:string) = f |> File.ReadLines |> Seq.map (fun x->x.Trim()) |> Seq.filter (fun x->x.Length > 0)
    {|Test=readFile test; Train=readFile train; Valid=readFile valid|}
