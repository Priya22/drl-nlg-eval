import torch

class Single:

    def __init__(self):
        pass

    def get_batch(self, x, vocab, device):
        go_x, x_eos = [], []
        max_len = max([len(s) for s in x])
        for s in x:
            s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
            padding = [vocab.pad] * (max_len - len(s))
            go_x.append([vocab.go] + s_idx + padding)
            x_eos.append(s_idx + [vocab.eos] + padding)
        return torch.LongTensor(go_x).t().contiguous().to(device), \
            torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

    def get_batches(self, data, vocab, batch_size, device):
        order = range(len(data))
        #z = sorted(zip(order, data), key=lambda i: len(i[1]))
        #order, data = zip(*z)

        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size): #and len(data[j]) == len(data[i]):
                j += 1
            batches.append(self.get_batch(data[i: j], vocab, device))
            i = j
        return batches, order

class Labelled:
    def __init__(self):
        pass

    def get_batch(self, x, y, vocab, device):
        go_x = []
        seq_lens = []
        #go_x2, x_eos2 = [], []

        max_len = max([len(s) for s in x])
        #print("Max Len: ", max_len)
        for s in x:
            s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
            padding = [vocab.pad] * (max_len - len(s))
            go_x.append([vocab.go] + s_idx + padding)
            seq_lens.append(len(s))
            #format y?

        return torch.LongTensor(go_x).t().contiguous().to(device), \
            seq_lens, \
            torch.LongTensor(y).t().contiguous().to(device) # time * batch

    def get_batches(self, data, labels, vocab, batch_size, device):

        order = range(len(data))
        #z = sorted(zip(order, data, labels), key=lambda i: len(i[1]))
        #order, data, labels = zip(*z)
       
        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size): #and len(data[0][j]) == len(data[0][i]):
                j += 1
            batches.append(self.get_batch(data[i: j], labels[i:j], vocab, device))
            i = j

        return batches, order

class MultLabelled:
    def __init__(self):
        pass

    def get_batch(self, x, yf, yc, vocab, device):
        go_x, x_eos, seq_lens = [], [], []
        max_len = max([len(s) for s in x])
        for s in x:
            s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
            padding = [vocab.pad] * (max_len - len(s))
            go_x.append([vocab.go] + s_idx + padding)
            x_eos.append(s_idx + [vocab.eos] + padding)
            seq_lens.append(len(s))

        return torch.LongTensor(go_x).t().contiguous().to(device), \
            torch.LongTensor(x_eos).t().contiguous().to(device), \
                seq_lens,\
                torch.LongTensor(yf).t().contiguous().to(device), \
                    torch.LongTensor(yc).t().contiguous().to(device) # time * batch

    def get_batches(self, data, classes, labels, vocab, batch_size, device):

        order = range(len(data))
        #z = sorted(zip(order, data, classes, labels), key=lambda i: len(i[1]))
        #order, data, classes, labels = zip(*z)
       
        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size): #and len(data[0][j]) == len(data[0][i]):
                j += 1
            batches.append(self.get_batch(data[i: j], classes[i:j], labels[i:j], vocab, device))
            i = j

        return batches, order

class BertBatch:
    def __init__(self):
        pass

    def get_batch(self, x, y, vocab, device):
        # go_x = []
        # #go_x2, x_eos2 = [], []

        # max_len = max([len(s) for s in x])
        # #print("Max Len: ", max_len)
        # for s in x:
        #     s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        #     padding = [vocab.pad] * (max_len - len(s))
        #     go_x.append([vocab.go] + s_idx + padding)
        #     #format y?

        return [' '.join(s) for s in x], \
            torch.LongTensor(y).t().contiguous().to(device) # time * batch

    def get_batches(self, data, labels, vocab, batch_size, device):

        order = range(len(data))
        z = sorted(zip(order, data, labels), key=lambda i: len(i[1]))
        order, data, labels = zip(*z)
       
        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size): #and len(data[0][j]) == len(data[0][i]):
                j += 1
            batches.append(self.get_batch(data[i: j], labels[i:j], vocab, device))
            i = j

        return batches, order


class Paraphrase:
    def __init__(self):
        pass

    def get_batch(self, x, y, vocab, device):
        go_x1, x_eos1, seq_lens1 = [], [], []
        go_x2, x_eos2, seq_lens2 = [], [], []

        max_len = max([len(s) for s in x[0]] + [len(s) for s in x[1]])
        #print("Max Len: ", max_len)
        for s1, s2 in zip(x[0], x[1]):
            s1_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s1]
            s2_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s2]

            padding1 = [vocab.pad] * (max_len - len(s1))
            padding2 = [vocab.pad] * (max_len - len(s2))

            go_x1.append([vocab.go] + s1_idx + padding1)
            x_eos1.append(s1_idx + [vocab.eos] + padding1)
            seq_lens1.append(len(s1))

            go_x2.append([vocab.go] + s2_idx + padding2)
            x_eos2.append(s2_idx + [vocab.eos] + padding2)
            seq_lens2.append(len(s2))

            #format y?

        return torch.LongTensor(go_x1).t().contiguous().to(device), \
            torch.LongTensor(x_eos1).t().contiguous().to(device), \
                seq_lens1,\
                torch.LongTensor(y[0]).t().contiguous().to(device), \
                torch.LongTensor(go_x2).t().contiguous().to(device), \
            torch.LongTensor(x_eos2).t().contiguous().to(device), \
                seq_lens2,\
                torch.LongTensor(y[1]).t().contiguous().to(device) # time * batch

    def get_batches(self, data, labels, vocab, batch_size, device):

        order = range(len(data[0]))
        #z = sorted(zip(order, data[0], data[1], labels[0], labels[1]), key=lambda i: len(i[1]))
        #order, data[0], data[1], labels[0], labels[1] = zip(*z)
        sorted_data = [data[0], data[1]]
        sorted_labels = [labels[0], labels[1]]

        #inputs1, targets1, inputs2, targets2, classes1, classes2  = [], [],[],[],[],[]
        batches = []
        i = 0
        while i < len(sorted_data[0]):
            j = i
            while j < min(len(sorted_data[0]), i+batch_size): #and len(data[0][j]) == len(data[0][i]):
                j += 1
            batches.append(self.get_batch([sorted_data[0][i: j], sorted_data[1][i:j]], [sorted_labels[0][i:j], sorted_labels[1][i:j]], vocab, device))
            i = j

        return batches, order
