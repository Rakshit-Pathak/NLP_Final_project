import numpy as np
def onehot_pos(pos):
    code = np.zeros(16)
    if pos == 'ADJ':
        code[0] = 1
        return code
    elif pos == 'ADP':
        code[1] = 1
        return code
    elif pos == 'ADV':
        code[2] = 1
        return code
    elif pos == 'AUX':
        code[3] = 1
        return code
    elif pos == 'CCONJ':
        code[4] = 1
        return code
    elif pos == 'DET':
        code[5] = 1
        return code
    elif pos == 'INTJ':
        code[6] = 1
        return code
    elif pos == 'NOUN':
        code[7] = 1
        return code
    elif pos == 'NUM':
        code[8] = 1
        return code
    elif pos == 'PART':
        code[9] = 1
        return code
    elif pos == 'PRON':
        code[10] = 1
        return code
    elif pos == 'PROPN':
        code[11] = 1
        return code
    elif pos == 'SCONJ':
        code[12] = 1
        return code
    elif pos == 'SYM':
        code[13] = 1
        return code
    elif pos == 'VERB':
        code[14] = 1
        return code
    elif pos == 'X':
        code[15] = 1
        return code
    else:
        return code
    