import numpy as np
def onehot_deprel(pos):
    code = np.zeros(47)
    if pos == 'acl':
        code[0] = 1
        return code
    elif pos == 'aacl:relcl':
        code[1] = 1
        return code
    elif pos == 'advcl':
        code[2] = 1
        return code
    elif pos == 'advmod':
        code[3] = 1
        return code
    elif pos == 'amod':
        code[4] = 1
        return code
    elif pos == 'appos':
        code[5] = 1
        return code
    elif pos == 'aux':
        code[6] = 1
        return code
    elif pos == 'aux:pass':
        code[7] = 1
        return code
    elif pos == 'case':
        code[8] = 1
        return code
    elif pos == 'cc':
        code[9] = 1
        return code
    elif pos == 'cc:preconj':
        code[10] = 1
        return code
    elif pos == 'ccomp':
        code[11] = 1
        return code
    elif pos == 'compound':
        code[12] = 1
        return code
    elif pos == 'compound:prt':
        code[13] = 1
        return code
    elif pos == 'conj':
        code[14] = 1
        return code
    elif pos == 'cop':
        code[15] = 1
        return code
    elif pos == 'csubj':
        code[16] = 1
        return code
    elif pos == 'csubj:pass':
        code[17] = 1
        return code
    elif pos == 'dep':
        code[18] = 1
        return code
    elif pos == 'det':
        code[19] = 1
        return code
    elif pos == 'det:predet':
        code[20] = 1
        return code
    elif pos == 'discourse':
        code[21] = 1
        return code
    elif pos == 'dislocated':
        code[22] = 1
        return code
    elif pos == 'expl':
        code[23] = 1
        return code
    elif pos == 'fixed':
        code[24] = 1
        return code
    elif pos == 'flat':
        code[25] = 1
        return code
    elif pos == 'flat:foreign':
        code[26] = 1
        return code
    elif pos == 'goeswith':
        code[27] = 1
        return code
    elif pos == 'iobj':
        code[28] = 1
        return code
    elif pos == 'list':
        code[29] = 1
        return code
    elif pos == 'mark':
        code[30] = 1
        return code
    elif pos == 'nmod':
        code[31] = 1
        return code
    elif pos == 'nmod:npmod':
        code[32] = 1
        return code
    elif pos == 'nmod:poss':
        code[33] = 1
        return code
    elif pos == 'nmod:tmod':
        code[34] = 1
        return code
    elif pos == 'nsubj':
        code[35] = 1
        return code
    elif pos == 'nsubj:pass':
        code[36] = 1
        return code
    elif pos == 'nummod':
        code[37] = 1
        return code
    elif pos == 'obj':
        code[38] = 1
        return code
    elif pos == 'obl':
        code[39] = 1
        return code
    elif pos == 'obl:npmod':
        code[40] = 1
        return code
    elif pos == 'obl:tmod':
        code[41] = 1
        return code
    elif pos == 'orphan':
        code[42] = 1
        return code
    elif pos == 'parataxis':
        code[43] = 1
        return code
    elif pos == 'reparandum':
        code[44] = 1
        return code
    elif pos == 'vocative':
        code[45] = 1
        return code
    elif pos == 'xcomp':
        code[46] = 1
        return code
    else:
        return code
    