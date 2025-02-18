
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from utils import get_embeddings_client
from tqdm import tqdm 


def embedding(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/embeddings.pkl"
    arguments = pd.read_csv(f"outputs/{dataset}/args.csv")
    embeddings = []
    model = config['embedding']['model']
    emb_model = get_embeddings_client(model)
    for i in tqdm(range(0, len(arguments), 1000)):
        args = arguments["argument"].tolist()[i: i + 1000]
        # embeds = OpenAIEmbeddings().embed_documents(args)
        embeds = emb_model.embed(
            input=args
        )
        batch_embeddings = [item.embedding for item in embeds.data]
        embeddings.extend(batch_embeddings)
        assert len(embeddings) == len(arguments), "Mismatch between embeddings and arguments count"
        # print('embeding info', embeddings[:10])
        # print('embeding info', len(embeddings))
        # print('arguments info', len(arguments))
    df = pd.DataFrame(
        [
            {"arg-id": arguments.iloc[i]["arg-id"], "embedding": e}
            for i, e in enumerate(embeddings)
        ]
    )
    print('embeding info', df.head())
    df.to_pickle(path)
