import torch
import torch.nn.functional as F
from typing import List, Union


class Bigram:
    """
    Expects a .txt file as input, containing for each row in the
    file, a datapoint.
    """

    def __init__(self, file, smoothing=0):
        self.file: str = file
        self.smoothing = smoothing

        self.chars = None
        self.s2i = None
        self.i2s = None

        self.N = None
        self.P = None
        self.one = None

        self._parse_file()

    def _parse_file(self):
        words = open(self.file, 'r').read().splitlines()

        self.chars = ['.'] + sorted(list(set(''.join(words))))
        self.s2i = {s: i for i, s in enumerate(self.chars)}
        self.i2s = {i: s for i, s in enumerate(self.chars)}

        n = len(self.chars)
        self.N = torch.zeros((n, n), dtype=torch.int32)
        self.one = torch.ones_like(self.N).float()
        self.one /= self.one.sum(1, keepdim=True)

        for w in words:
            ch = ['.'] + list(w) + ['.']
            for c1, c2 in zip(ch, ch[1:]):
                i = self.s2i[c1]
                j = self.s2i[c2]
                self.N[i, j] += 1

        # Smoothing
        self.N += self.smoothing

        self.P = self.N / self.N.sum(1, keepdim=True)

    def generate(self, n: int, random: bool = False):
        """
        With random set to True, returns an output generated from
        a uniform distribution.
        """
        names = {}
        for i in range(n):

            g = torch.Generator().manual_seed(2147483647)

            word = '.'
            idx = self.s2i[word]
            count = 0
            while True:
                p = self.P[idx] if not random else self.one[idx].float()

                idx = torch.multinomial(p, num_samples=1, replacement=True).item()
                cha = self.i2s[idx]
                word += cha
                if cha == '.' or count == 100:
                    break
                count += 1
            names[i] = word
        return names

    def _get_cond_prob(self, char: str, given: str):
        i = self.s2i[given]
        j = self.s2i[char]
        cond_prob = self.P[i, j]
        return cond_prob

    def log_likelihood(self, words, random=False):
        if type(words) == str:
            words = [words]
        n = 0
        log_likelihood = 0.0
        for w in words:
            ch = ['.'] + list(w) + ['.']
            for c1, c2 in zip(ch, ch[1:]):
                i = self.s2i[c1]
                j = self.s2i[c2]
                logprob = torch.log(self.P[i, j]) if not random else torch.log(self.one[i, j])
                log_likelihood += logprob
                n += 1
        return log_likelihood / n

    def __repr__(self):
        i, j = self.N.shape
        return f"Bigram-model of token size: ({i}x{j}), smoothing: {self.smoothing}"


class BigramNeuralNet:

    def __init__(self, file):
        self.file: str = file

        self.chars = None
        self.s2i = None
        self.i2s = None

        self.W: Union[torch.Tensor, None] = None
        self.xs = None
        self.xy = None

        self._parse_file()

    def _parse_file(self):
        words = open(self.file, 'r').read().splitlines()

        xs, ys = [], []

        self.chars = ['.'] + sorted(list(set(''.join(words))))
        self.s2i = {s: i for i, s in enumerate(self.chars)}
        self.i2s = {i: s for i, s in enumerate(self.chars)}

        n = len(self.chars)
        self.W = torch.randn((n, n), requires_grad=True)

        for w in words:
            chs = ['.'] + list(w) + ['w']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.s2i[ch1]
                ix2 = self.s2i[ch2]

                xs.append(ix1)
                ys.append(ix2)

        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def train(self, epochs=10, lr=10):

        for _ in range(epochs):
            xenc = F.one_hot(self.xs, num_classes=27).float();
            out = xenc @ self.W
            a = torch.exp(out)
            prob = a / a.sum(1, keepdim=True)
            nlls = -prob[torch.arange(228146), self.ys].log().mean()
            print(nlls)

            # Backward
            self.W.grad = None
            nlls.backward()
            self.W.data -= lr * self.W.grad

    def generate(self, n: int):
        names = {}
        for i in range(n):
            word = '.'
            idx = self.s2i[word]
            count = 0
            while True:
                xenc = F.one_hot(torch.tensor(idx), num_classes=27).float();
                out = xenc @ self.W
                a = torch.exp(out)
                pdist = a / a.sum()

                idx = torch.multinomial(pdist, num_samples=1, replacement=True).item()
                cha = self.i2s[idx]
                word += cha
                if cha == '.' or count == 100:
                    break
                count += 1
            names[i] = word
        return names



