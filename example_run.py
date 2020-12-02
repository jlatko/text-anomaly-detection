from experiment import train_lm

if __name__ == '__main__':
    data_path = 'data'
    tags = 'parliament vs friends scale=0.2 bs=32 \n'
    # source = 'friends-corpus'
    # source = 'supreme-corpus'
    ood_source = 'friends-corpus'
    source = 'parliament-corpus'
    # source = 'IMDB Dataset.csv'
    split_sentences = True
    punct = False
    to_ascii = True
    min_len = 3
    max_len = 15
    test_size = 0.1
    text_field = 'text'
    batch_size = 32
    word_embedding_size = 50
    optimizer_kwargs = {
        'lr': 1e-3
    }
    n_epochs = 100
    print_every = 1
    subsample_rows = None  # for testing
    subsample_rows_ood = None
    min_freq = 1
    decode = False
    model_kwargs = {
        'set_other_to_random': False,
        'set_unk_to_random': True,
        'decode_with_embeddings': decode,  # [False, 'cosine', 'cdist']
        # 'p_word_dropout': 0.3,
        'max_sent_len': max_len,
        'freeze_embeddings': False,
        'rnn_dropout': 0.3,
        'mask_pad': True,
    }
    kl_kwargs = {
        'cycles': 4,
        'scale': 0.2
    }

    train_lm(source=source,
             batch_size=batch_size,
             word_embedding_size=word_embedding_size,
             model_kwargs=model_kwargs,
             optimizer_kwargs=optimizer_kwargs,
             kl_kwargs=kl_kwargs,
             n_epochs=n_epochs,
             print_every=print_every,
             split_sentences=split_sentences,
             punct=punct,
             to_ascii=to_ascii,
             min_freq=min_freq,
             min_len=min_len,
             max_len=max_len,
             test_size=test_size,
             text_field=text_field,
             subsample_rows=subsample_rows,
             data_path=data_path,
             ood_source=ood_source,
             subsample_rows_ood=subsample_rows_ood,
             tags=tags
             )
