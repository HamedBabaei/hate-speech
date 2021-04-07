from summarizer import TransformerSummarizer

tsm = TransformerSummarizer(transformer_type='Bert', transformer_model_key = 'bert-base-uncased')

def summarize(X, no=1, num_sentences=15):
    summary = tsm(X, num_sentences=15)
    summaries = summary.split('<TWEET>')
    lenght = len(summaries)
    size = int(lenght/no)
    sums = []
    for i in range(0, size):
        sums.append(summaries[i*size:(i+1)*size])
    return sums
