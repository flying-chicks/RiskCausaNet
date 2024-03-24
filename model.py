# encoding: utf-8

import warnings

warnings.filterwarnings("ignore")
from dgl.nn import GraphConv
from transformers import AutoModelForCausalLM
import os
import torch
from torch import nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TextEncoder(nn.Module):
    # def __init__(self, llm_path):
    def __init__(self):
        super().__init__()
        llm = AutoModelForCausalLM.from_pretrained('trained_model').to('cuda:0')
        self.model = llm.model
        self.lm_head = llm.lm_head

    def forward(self, input_token_ids):
        outputs = self.model(input_token_ids)
        return outputs[0]


def unbatch_node_embeddings(graph, node_embeddings_2d):
    '''
    因为GraphEncoder得到的结果是batch_size个dgl图的节点特征，其内容是node_embeddings_2d
    目的是分离graph中几个小图的node_embeddings，并将他们转为新的维度[batch_size,max_num_node,dim=1024]
    graph 是由dgl小图合并的大图
    node_embeddings_2d  维度是[node_sum,dim=1024]
    '''
    # batch_num_nodes是一个tensor，例如tensor（[4,3]）表示7个节点，第一个图节点从0到3，第二个图节点从4到6
    device = graph.device
    batch_num_nodes = graph.batch_num_nodes()
    # batch_num_nodes例如tensor（[4,3]）--》tensor（[0，4,3]）
    batch_num_nodes = torch.cat([torch.LongTensor([0]).to(device), batch_num_nodes])
    # 例如index=tensor（[0,4,7]）
    index = batch_num_nodes.cumsum(dim=0)
    # 找batch_num_nodes中最大的数，例如4
    max_num_node = batch_num_nodes.max().item()

    node_embeddings_3d = []
    for i in range(len(index) - 1):
        # batch_embeddings是索引node_embeddings_2d中对应小图的节点，维度是[node_num,dim=1024]
        batch_embeddings = node_embeddings_2d[index[i]: index[i + 1], :]

        diff = max_num_node - batch_embeddings.size(0)
        # 如果节点数量不够，就用0补全它
        if diff > 0:
            batch_embeddings = torch.cat([batch_embeddings,
                                          torch.zeros(diff, batch_embeddings.size(1)).to(device)],
                                         dim=0)

        batch_embeddings = batch_embeddings.unsqueeze(0)
        node_embeddings_3d.append(batch_embeddings)

    # [batch_size,max_num_node,dim=1024]
    return torch.cat(node_embeddings_3d, dim=0)


class GraphEncoder(nn.Module):
    def __init__(self, in_feat, n_layers, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feat, hidden_size))
        self.layers.append(nn.ReLU())

        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        out_layer = GraphConv(hidden_size, out_size)
        self.layers.append(out_layer)

    # node_feats : (num_nodes, feature_dim=3072)
    def forward(self, graph, node_feats):
        for layer in self.layers:
            if isinstance(layer, GraphConv):

                node_feats = layer(graph, node_feats)

            else:
                node_feats = layer(node_feats)
        # print("node_feats_2有无nan", torch.isnan(node_feats).any())
        # node_feats维度[node_num,out_size=1024]，node_feats_3d维度[batch_size,max_num_node,dim=1024]
        node_feats_3d = unbatch_node_embeddings(graph, node_feats)

        return node_feats_3d  # [batch_size,max_num_node,dim=1024]


class JointReasoning(nn.Module):
    def __init__(self,
                 g_in_feat,
                 g_n_layers,
                 g_hidden_size,
                 g_out_size,
                 n_head,
                 # llm_path,
                 # graph_path=None,
                 # attn_path=None
                 ):
        super().__init__()
        self.device = torch.device('cuda:0')

        # self.text_encoder = TextEncoder(llm_path).to(self.device)
        self.text_encoder = TextEncoder().to(self.device)
        self.graph_encoder = GraphEncoder(g_in_feat, g_n_layers, g_hidden_size, g_out_size).to(self.device)
        # if graph_path is not None:
        #     self.graph_encoder.load_state_dict(torch.load(graph_path))

        # 必须保证text的维度和node_embedding的维度一样
        self.attention_layer = nn.MultiheadAttention(embed_dim=g_out_size,
                                                     num_heads=n_head,
                                                     dropout=0.1,
                                                     batch_first=True).to(self.device)
        # if attn_path is not None:
        #     self.attention_layer.load_state_dict(torch.load(attn_path))
        self.info_nce_fn = InfoNCE()
        self.cross_entropy_fn = nn.CrossEntropyLoss()

    def calc_nce_label(self, token_ids, node_feature, beta):
        token_ids = token_ids.to(self.device)
        token_embeddings = self.text_encoder(token_ids).contiguous()  # [token_len+start,3072]

        node_feature = node_feature.to(torch.float32).to(self.device)  # [node,3072]
        node_feature = torch.unsqueeze(node_feature, dim=0)

        node_embeddings = F.normalize(node_feature, p=2, dim=-1)
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
        cos_sim = torch.bmm(node_embeddings, token_embeddings.transpose(-2, -1))
        label = torch.where(cos_sim > beta, torch.tensor(1), torch.tensor(0))
        return label.to(torch.float32)

    def forward(self, token_ids, graph, node_feature, nce_labels, alpha=0.01):
        token_ids = token_ids.to(self.device)
        graph = graph.to(self.device)
        node_feature = node_feature.to(torch.float32).to(self.device)
        token_embeddings = self.text_encoder(token_ids)
        node_embeddings = self.graph_encoder(graph, node_feature)

        att_embeddings, att_weights = self.attention_layer(token_embeddings, node_embeddings, node_embeddings,
                                                           average_attn_weights=False)

        hidden_states = (1 - alpha) * token_embeddings + alpha * att_embeddings
        logit = self.text_encoder.lm_head(hidden_states)

        shift_logit = logit[:, :-1, :].contiguous()
        cross_entropy = self.cross_entropy_fn(shift_logit.view(-1, shift_logit.size(-1)),
                                              token_ids[:, 1:].contiguous().view(-1))

        info_nce = self.info_nce_fn(node_embeddings, token_embeddings, nce_labels)
        total_loss = cross_entropy + info_nce
        return shift_logit, att_weights, total_loss

    def generate(self, token_ids, graph, node_feature, alpha=0.01):
        token_ids = token_ids.to(self.device)
        graph = graph.to(self.device)
        node_feature = node_feature.to(torch.float32).to(self.device)
        token_embeddings = self.text_encoder(token_ids)
        node_embeddings = self.graph_encoder(graph, node_feature)

        att_embeddings, att_weights = self.attention_layer(token_embeddings, node_embeddings, node_embeddings,
                                                           average_attn_weights=False)

        hidden_states = (1 - alpha) * token_embeddings + alpha * att_embeddings
        logit = self.text_encoder.lm_head(hidden_states)

        return logit


def generate(tokenizer, model, prompt, graph, node_feature, max_length=50, temperature=1.0, top_k=3):
    model = model.to(torch.device('cuda:0'))

    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    graph = graph.to(torch.device('cuda:0'))
    node_feature = node_feature.to(torch.device('cuda:0'))
    input_ids = input_ids.to(torch.device('cuda:0'))

    # 初始化生成的文本为输入文本
    generated = input_ids

    with torch.no_grad():  # 不计算梯度
        for _ in range(max_length):
            predictions = model.generate(generated, graph, node_feature)

            # 采用最后一个时间步的预测结果
            next_token_logits = predictions[:, -1, :] / temperature

            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
            # 将新生成的词添加到生成的文本中
            generated = torch.cat((generated, next_token), dim=-1)

            # 检查是否生成了结束符
            if next_token in tokenizer.encode(tokenizer.eos_token):
                break
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return generated_text


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def info_nce(self, query, positive_key, labels, temperature=0.1, reduction='mean'):
        query, positive_key = self.normalize(query, positive_key)
        logits = query @ self.transpose(positive_key)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels, reduction=reduction)
        return loss

    def forward(self, query, positive_key, labels):
        return self.info_nce(query, positive_key, labels,
                             temperature=self.temperature,
                             reduction=self.reduction)

