from os import listdir, mkdir
from os.path import isdir

from nltk import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration


def get_summary(doc: str, model_bart, tokenizer_bart) -> str:
    if len(doc.split()) < 2:
        return ''

    inputs = tokenizer_bart.encode(doc, max_length=tokenizer_bart.model_max_length,
                                   truncation=True,
                                   truncation_strategy='do_not_truncate',
                                   return_tensors='pt')

    print(f' Sub-Tokenized input length: {len(inputs[0])}')

    outputs = model_bart.generate(inputs, do_sample=False, num_beams=5, early_stopping=False)

    summary = [tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]

    print(f'Summary Generated | Output length: {len(summary[0].split())} | '
          f'Sub-Tokenized output length: {len(outputs[0])}')

    return summary[0]


def create_summaries(src_path: str, dest_path: str, model_bart, tokenizer_bart):
    if not isdir(dest_path):
        mkdir(dest_path)

    count = 1

    for doc in listdir(src_path):
        if doc not in listdir(dest_path):
            if not doc.startswith('.'):
                with open(src_path + doc, 'r') as f:
                    text = f.read()
                    print(f'\n\nDoc#{count}')
                    print(f'\nGenerating summary for {doc} | Input length: {len(text.split())} ', end='|')
                    summary = get_summary(text, model_bart, tokenizer_bart)
                    summary = '\n\n'.join([token for token in sent_tokenize(summary)])

                with open(dest_path + doc, 'w') as g:
                    g.write(summary)
        count += 1


def generate_bart_summ():
    model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    for section_folder in listdir('Data/Input-BART'):
        if not section_folder.startswith('.'):
            print(f'\n{section_folder}::\n')
            create_summaries(src_path=f'Data/Input-BART/{section_folder}/',
                             dest_path=f'Data/Section-wise-summaries/bart_{section_folder}/', model_bart=model_bart,
                             tokenizer_bart=tokenizer_bart)
