# based on https://github.com/MhLiao/MaskTextSpotter/blob/master/evaluation/icdar2015/e2e/weighted_editdistance.py
# MIT license

import numpy as np

def weighted_edit_distance(word1, word2, scores):
    m = len(word1)
    n = len(word2)
    dp = np.zeros((n + 1, m + 1))
    dp[0, :] = np.arange(m + 1)
    dp[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete_cost = ed_delete_cost(j - 1, word1, scores)
            insert_cost = ed_insert_cost(j - 1, word1, scores)
            replace_cost = ed_replace_cost(j - 1, i - 1, word1, word2, scores) if word1[j - 1] != word2[i - 1] else 0
            dp[i][j] = min(dp[i - 1][j] + insert_cost, dp[i][j - 1] + delete_cost, dp[i - 1][j - 1] + replace_cost)

    return dp[n][m]

def ed_delete_cost(j, word1, scores):
    c = char2num(word1[j])
    return scores[c][j]


def ed_insert_cost(i, word1, scores):
    if i < len(word1) - 1:
        c1 = char2num(word1[i])
        c2 = char2num(word1[i + 1])
        return (scores[c1][i] + scores[c2][i+1]) / 2
    else:
        c1 = char2num(word1[i])
        return scores[c1][i]


def ed_replace_cost(i, j, word1, word2, scores):
    c1 = char2num(word1[i])
    c2 = char2num(word2[j])
    return max(1 - scores[c2][i] / scores[c1][i] * 5, 0)

def char2num(char):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return alphabet.find(char.upper())
