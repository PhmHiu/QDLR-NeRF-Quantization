import math
from sklearn.cluster import KMeans
import re
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
from nerf import FlexibleNeRFModel

def get_weight_mats(net):
    weight_mats = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.weight', name, re.I)]
    return [mat[1].cpu() for mat in weight_mats]

def get_bias_vecs(net):
    bias_vecs = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.bias', name, re.I)]
    return [bias[1].cpu() for bias in bias_vecs]

def ints_to_bits_to_bytes(all_ints,n_bits):
    f_str = '#0'+str(n_bits+2)+'b'
    bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
    n_bytes = len(bit_string)//8
    the_leftover = len(bit_string)%8>0
    if the_leftover:
        n_bytes+=1
    the_bytes = bytearray()
    for b in range(n_bytes):
        bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
        the_bytes.append(int(bin_val,2))
    return the_bytes,the_leftover

class SimpleMap(dict):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(SimpleMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

class Encoder:
    def __init__(self,net,config):
        self.net = net
        self.config = config

    def kmeans_quantization(self, w):
        weight_feat = w.view(-1).unsqueeze(1).numpy()
        labels = self.kmeans.predict(weight_feat)
        return labels.tolist()
    
    def update_codebook(self, w_list):
        print('Updating codebook...')
        w_feat = []
        for w in w_list:
            w_feat.append(w.view(-1).unsqueeze(1).numpy())
        w_feat = np.concatenate(w_feat, axis=0)
        print('Number of weights counted:', w_feat.shape[0])
        self.kmeans = self.kmeans.fit(w_feat)
    
    def draw_distribution(self, n_bins=1000, dpi=10):
        w_list = get_weight_mats(self.net)
        w_feat = []
        for w in w_list:
            w_feat.append(w.view(-1).numpy())
        w_feat = np.concatenate(w_feat, axis=0)
        plt.hist(w_feat, bins=n_bins)
        plt.savefig("Before_convert.png", dpi=dpi)
    
    def encode(self,filename, n_centroids=256, lower_bound=-3, upper_bound=3):
        self.kmeans = KMeans(n_clusters=n_centroids, n_init=5)
        num_layers = self.config['num_layers']
        hidden_size = self.config['hidden_size']
        skip_connect_every = self.config['skip_connect_every']
        num_encoding_fn_xyz = self.config['num_encoding_fn_xyz']
        num_encoding_fn_dir = self.config['num_encoding_fn_dir']

        weight_mats = get_weight_mats(self.net)
        bias_vecs = get_bias_vecs(self.net)
        self.update_codebook(weight_mats)
        self.centers = self.kmeans.cluster_centers_.reshape(n_centroids).tolist()
        file = open(filename, 'wb')

        # header: number of layers
        header = file.write(struct.pack('B', num_layers))
        # header: hidden_size
        header += file.write(struct.pack('B', hidden_size))
        # header: skip connect in how many layers
        header += file.write(struct.pack('B', skip_connect_every))
        # header: num_encoding_fn_xyz
        header += file.write(struct.pack('B', num_encoding_fn_xyz))
        # header: num_encoding_fn_dir
        header += file.write(struct.pack('B', num_encoding_fn_dir))
        # header: number of centroid
        header += file.write(struct.pack('H', n_centroids))

        weight = 0
        # weights: store centroid
        centroid_format = ''.join(['f' for _ in range(len(self.centers))])
        weight += file.write(struct.pack(centroid_format, *self.centers))
        # weights: map weight to cluster then store weight
        for weight_mat, bias_vec in zip(weight_mats,bias_vecs):
            labels = self.kmeans_quantization(weight_mat)
            # weights
            weight_bin, is_leftover = ints_to_bits_to_bytes(labels, math.ceil(math.log2(n_centroids)))
            weight += file.write(weight_bin)

            # encode non-pow-2 as 16-bit integer
            if math.ceil(math.log2(n_centroids))%8 != 0:
                weight += file.write(struct.pack('I', labels[-1]))

            # bias
            b = bias_vec.view(-1).tolist()
            b_format = ''.join(['e' for _ in range(len(b))])
            weight += file.write(struct.pack(b_format, *b))

        file.flush()
        file.close()
        print("Write completed!")

class Decoder:
    def __init__(self):
        pass

    def draw_distribution(self, n_bins=1000, dpi=10):
        w_list = get_weight_mats(self.net)
        w_feat = []
        for w in w_list:
            w_feat.append(w.view(-1).numpy())
        w_feat = np.concatenate(w_feat, axis=0)
        plt.hist(w_feat, bins=n_bins)
        plt.savefig("After_convert.png", dpi=dpi)
    
    def decode(self, filename):
        #weight_mats = get_weight_mats(self.net)
        #bias_vecs = get_bias_vecs(self.net)

        file = open(filename,'rb')

        # header: number of layers
        self.num_layers = struct.unpack('B', file.read(1))[0]
        # header: hidden_size
        self.hidden_size = struct.unpack('B', file.read(1))[0]
        # header: skip_connect_every
        self.skip_connect_every = struct.unpack('B', file.read(1))[0]
        # header: num_encoding_fn_xyz
        self.num_encoding_fn_xyz = struct.unpack('B', file.read(1))[0]
        dim_xyz = 3 + 2*3*self.num_encoding_fn_xyz
        # header: num_encoding_fn_dir
        self.num_encoding_fn_dir = struct.unpack('B', file.read(1))[0]
        dim_dir = 3 + 2*3*self.num_encoding_fn_dir
        # header: n_centroid
        self.n_centroid = struct.unpack('H', file.read(2))[0]
        self.n_bits = math.ceil(math.log2(self.n_centroid))
        # Build model from header config
        model = FlexibleNeRFModel(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            skip_connect_every=self.skip_connect_every,
            num_encoding_fn_xyz=self.num_encoding_fn_xyz,
            num_encoding_fn_dir=self.num_encoding_fn_dir
        )
        
        # weights: centroid
        self.centers = torch.FloatTensor(struct.unpack(''.join(['f' for _ in range(self.n_centroid)]), file.read(4*(self.n_centroid))))

        # first layer: matrix and bias
        # weights
        n_weights = dim_xyz*self.hidden_size
        weight_size = (n_weights*self.n_bits)//8
        if (n_weights*self.n_bits)%8 != 0:
            weight_size+=1
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])
        if self.n_bits%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
        w_quant = self.centers[w_inds]
        # bias
        b_format = ''.join(['e' for _ in range(self.hidden_size)])
        bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*self.hidden_size)))

        all_ws = [w_quant]
        all_bs = [bias]

        # middle layers
        for ldx in range(self.num_layers - 1):
            if ldx % self.skip_connect_every == 0 and ldx > 0 and ldx != self.num_layers - 1:
                # weights
                n_weights = (dim_xyz + self.hidden_size) * self.hidden_size
            else:
                # weights
                n_weights = self.hidden_size * self.hidden_size
            weight_size = (n_weights*self.n_bits)//8
            if (n_weights*self.n_bits)%8 != 0:
                weight_size+=1
            inds = file.read(weight_size)
            bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
            w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])

            if self.n_bits%8 != 0:
                next_bytes = file.read(4)
                w_inds[-1] = struct.unpack('I', next_bytes)[0]

            # bias
            b_format = ''.join(['e' for _ in range(self.hidden_size)])
            bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*self.hidden_size)))

            w_quant = self.centers[w_inds]
            all_ws.append(w_quant)
            all_bs.append(bias)

        # directional layer: matrix and bias
        # weights
        n_weights = (dim_dir + self.hidden_size) * (self.hidden_size // 2)
        weight_size = (n_weights*self.n_bits)//8
        if (n_weights*self.n_bits)%8 != 0:
            weight_size+=1
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])
        if self.n_bits%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
        w_quant = self.centers[w_inds]
        # bias
        b_format = ''.join(['e' for _ in range(self.hidden_size // 2)])
        bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*(self.hidden_size//2))))

        all_ws.append(w_quant)
        all_bs.append(bias)

        # alpha layer: matrix and bias
        # weights
        n_weights = self.hidden_size
        weight_size = (n_weights*self.n_bits)//8
        if (n_weights*self.n_bits)%8 != 0:
            weight_size+=1
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])
        if self.n_bits%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
        w_quant = self.centers[w_inds]
        # bias
        b_format = ''.join(['e' for _ in range(1)])
        bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*1)))

        all_ws.append(w_quant)
        all_bs.append(bias)

        # rgb layer: matrix and bias
        # weights
        n_weights = (self.hidden_size // 2) * 3
        weight_size = (n_weights*self.n_bits)//8
        if (n_weights*self.n_bits)%8 != 0:
            weight_size+=1
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])
        if self.n_bits%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
        w_quant = self.centers[w_inds]
        # bias
        b_format = ''.join(['e' for _ in range(3)])
        bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*3)))

        all_ws.append(w_quant)
        all_bs.append(bias)

        # feature layer: matrix and bias
        # weights
        n_weights = self.hidden_size * self.hidden_size
        weight_size = (n_weights*self.n_bits)//8
        if (n_weights*self.n_bits)%8 != 0:
            weight_size+=1
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])
        if self.n_bits%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
        w_quant = self.centers[w_inds]
        # bias
        b_format = ''.join(['e' for _ in range(self.hidden_size)])
        bias = torch.FloatTensor(struct.unpack(b_format, file.read(2*self.hidden_size)))

        all_ws.append(w_quant)
        all_bs.append(bias)

        wdx,bdx=0,0
        for name, parameters in model.named_parameters():
            if re.match(r'.*.weight', name, re.I):
                w_shape = parameters.data.shape
                parameters.data = all_ws[wdx].view(w_shape)
                wdx+=1

            if re.match(r'.*.bias', name, re.I):
                b_shape = parameters.data.shape
                parameters.data = all_bs[bdx].view(b_shape)
                bdx+=1
        self.net = model
        return model