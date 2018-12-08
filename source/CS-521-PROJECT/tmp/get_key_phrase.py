## helper function to extract compound noun phrase or noun adjective
import spacy, en_core_web_sm

def get_key_phrase(sent):
    nlp=en_core_web_sm.load()
    parsed=nlp(sent)
    compound_nn_adj, reg_nn_adj, props=[],[],[]
    compound_list=[tok for tok in parsed if tok.dep_=='compound']
    compound_list=[nc for nc in compound_list if nc.i==0 or parsed[nc.i-1].dep_!='compound']
    if len(compound_list)!=0:
        for elm in compound_list:
            pair_1, pair_2=(False, False)
            noun=parsed[elm.i:elm.head.i+1]
            pair_1=noun
            if noun.root.dep_=='nsubj':
                adj_list=[rt for rt in noun.root.head.right if rt.pos_=='ADJ']
                if adj_list:
                    pair_2=adj_list[0]
            if noun.root.dep_ == 'dobj':
                verb_root = [vb for vb in noun.root.ancestors if vb.pos_ == 'VERB']
                if verb_root:
                    pair_2 = verb_root[0]
            if pair_1 and pair_2:
                compounds_nn_pairs.append(pair_1, pair_2)
    else:
        for i, tok in enumerate(parsed):
            if tok.pos_ not in ('NOUN','PRON'):
                continue
            for j in range(i+1, len(parsed)):
                if parsed[j].pos_=='ADJ':
                    reg_nn_adj.append(tok, parsed[j])
                if parsed[j].pos_=='ADP':
                    props.append(tok, parsed[j])
    return compound_nn_adj, reg_nn_adj, props

## how to use the function
phrase_corpus=[]
for i in range(0, len(dataset)):
    parsed=nlp(dataset['statement'][i])
    res=get_key_phrase(parsed)
    phrase_corpus.append(res)

