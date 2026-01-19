# 包引入
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer

def generate_code_embeddings(code_data, model_name):
    """
    加载代码，生成嵌入
    """
    
    # 加载预训练模型
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # 生成嵌入
    print("Generating embeddings...")
    embeddings = model.encode(code_data, show_progress_bar=True).tolist()
    
    return embeddings

class CrossAttention(nn.Module):
    '''
    模型架构(必须导入，不必修改)
    '''
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # query, key, value: [batch_size, seq_len, embed_dim]
        attn_output, _ = self.attention(query, key, value)
        return self.norm(attn_output + query)  # Add & Norm

class AblationModelWithWeights(nn.Module):
    '''
    模型整体架构(必须导入，不必修改)
    '''
    def __init__(self, 
                 code_input_dim, 
                 tabular_input_dim, 
                 gcn_hidden_dim, 
                 gcn_output_dim, 
                 embed_dim=32,
                 use_code=True, 
                 use_tabular=True, 
                 use_graph=True):
        super(AblationModelWithWeights, self).__init__()

        # 是否使用各分支
        self.use_code = use_code
        self.use_tabular = use_tabular
        self.use_graph = use_graph

        # Code Embedding 分支
        if self.use_code:
            self.code_branch = nn.Sequential(
                nn.Linear(code_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, embed_dim)
            )

        # Tabular 数据分支
        if self.use_tabular:
            self.tabular_branch = nn.Sequential(
                nn.Linear(tabular_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, embed_dim),
                nn.ReLU()
            )

        # GCN 分支
        if self.use_graph:
            self.gcn1 = GCNConv(32, gcn_hidden_dim)
            self.bn1 = nn.BatchNorm1d(gcn_hidden_dim)
            self.gcn2 = GCNConv(gcn_hidden_dim, gcn_output_dim)
            self.bn2 = nn.BatchNorm1d(gcn_output_dim)

        # 交叉注意力模块
        self.cross_attention_1 = CrossAttention(embed_dim)
        self.cross_attention_2 = CrossAttention(embed_dim)

        # 模态权重参数（可学习）
        self.modal_weights = nn.Parameter(torch.ones(3))  # 初始化为 1，表示每个模态的初始权重相等

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(
                (embed_dim if self.use_code else 0) +
                (embed_dim if self.use_tabular else 0) +
                (gcn_output_dim if self.use_graph else 0), 
                128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 输出4个标签
        )

    def forward(self, code_data, tabular_data, graph_data):
        fusion_input = []
        modal_contributions = []

        # Code Embedding 分支
        if self.use_code:
            code_out = self.code_branch(code_data)  # [batch_size, embed_dim]
            fusion_input.append(code_out)
            modal_contributions.append(self.modal_weights[0])  # 添加 code 模态的权重

        # Tabular 数据分支
        if self.use_tabular:
            tabular_out = self.tabular_branch(tabular_data)  # [batch_size, embed_dim]
            fusion_input.append(tabular_out)
            modal_contributions.append(self.modal_weights[1])  # 添加 tabular 模态的权重

        # GCN 分支
        if self.use_graph:
            x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
            x = F.relu(self.bn1(self.gcn1(x, edge_index)))
            x = F.relu(self.bn2(self.gcn2(x, edge_index)))
            x = global_mean_pool(x, batch)  # [batch_size, gcn_output_dim]
            fusion_input.append(x)
            modal_contributions.append(self.modal_weights[2])  # 添加 graph 模态的权重

        # 如果没有启用任何分支，直接抛出错误
        if len(fusion_input) == 0:
            raise ValueError("At least one of 'use_code', 'use_tabular', or 'use_graph' must be True.")

        # 对模态权重进行 softmax 标准化
        modal_contributions = F.softmax(torch.stack(modal_contributions), dim=0)  # [num_modalities]
        
        # 对每个模态的输出乘以对应的权重
        weighted_fusion_input = [
            modal_contributions[i] * fusion_input[i] for i in range(len(fusion_input))
        ]

        # 融合
        fusion_input = torch.cat(weighted_fusion_input, dim=1)  # 按最后一维拼接
        out = self.fusion_layer(fusion_input)  # [batch_size, 4]

        return out

def predict_model(config, model_path, code_embeddings, device):
    '''
    预测模型
    '''
    # 初始化模型
    model = AblationModelWithWeights(
        code_input_dim=768,  # 输入为预先生成的嵌入维度
        tabular_input_dim=16,
        gcn_hidden_dim=32,
        gcn_output_dim=16,
        embed_dim=32,
        **config
    ).to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 模态处理逻辑
    with torch.no_grad():
        # 直接处理整个 code_embeddings
        code_data = torch.tensor(code_embeddings).unsqueeze(0).to(device)  # 生成 [1, 768] 的张量
        tabular_data = torch.zeros(code_data.size(0), 16).to(device)  # 填充 tabular 数据为 (batch_size, 16)
        graph_data = Data(
            x=torch.zeros(code_data.size(0), 32).to(device),  # 填充节点特征为 32
            edge_index=torch.zeros(2, code_data.size(0)).to(device).long(),  # 填充边索引
            batch=torch.zeros(code_data.size(0), dtype=torch.long).to(device)  # 填充批次索引
        )
        # 模型推理
        outputs = model(code_data, tabular_data, graph_data)
    return outputs.cpu()

def main():
    # 训练设备部署
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    codebert-base需要安装

    1、安装huggingface-cli（使用镜像）
    pip install 'huggingface_hub[cli]' --index-url=https://mirrors.aliyun.com/pypi/simple

    2、下载codebert-base模型
    huggingface-cli download microsoft/codebert-base --local-dir 预存储本地位置

    '''

    # 接收输入的代码
    source_code= "__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B)\n{\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i = blockIdx.y * blockDim.y + threadIdx.y;\n\n\tif ((i < _PB_NI) && (j < _PB_NJ))\n\t{ \n\t\ttmp[i * NJ + j] = 0;\n\t\tint k;\n\t\tfor (k = 0; k < _PB_NK; k++)\n\t\t{\n\t\t\ttmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];\n\t\t}\n\t}\n}"

    # 替换为codebert-base本地存储位置
    embedding_model = "/home/pm/codebert-base"

    # 输入代码，生成嵌入
    code_embedding = generate_code_embeddings(source_code, embedding_model)

    # 存储所有预测结果
    all_predictions = []

    # 设置模态
    configs = [
        {'use_code': True, 'use_tabular': True, 'use_graph': True},  # 全部启用（Baseline）
        {'use_code': False, 'use_tabular': True, 'use_graph': True},  # 去掉 Code 分支
        {'use_code': True, 'use_tabular': False, 'use_graph': True},  # 去掉 Tabular 分支
        {'use_code': True, 'use_tabular': True, 'use_graph': False},  # 去掉 Graph 分支
        {'use_code': True, 'use_tabular': False, 'use_graph': False},  # 只有code 分支
    ]
    for config in configs:
        # 模型加载
        model_load_path = f"model/model_{str(config)}.pth"
        # 预测
        prediction=predict_model(config, model_load_path, code_embedding, device)
        all_predictions.append(prediction)
        
    last_prediction = all_predictions[-1]
    print(last_prediction)
    # 参数名称
    parameter_names = ["AMAT", "Time", "L1 Hit Rate", "L2 Hit Rate"]

    # 提取并打印每个预测的结果
    for idx, prediction in enumerate(all_predictions):
        print(f"Prediction {idx + 1}:")
        for param_idx, param_name in enumerate(parameter_names):
            print(f"  {param_name}: {prediction[0][param_idx]:.4f}")
        print()  # 空行分隔每个预测结果


def analyze_hip(input_code):
    # 训练设备部署
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    codebert-base需要安装

    1、安装huggingface-cli（使用镜像）
    pip install 'huggingface_hub[cli]' --index-url=https://mirrors.aliyun.com/pypi/simple

    2、下载codebert-base模型
    huggingface-cli download microsoft/codebert-base --local-dir 预存储本地位置

    '''

    # 接收输入的代码
    source_code = input_code

    # 替换为codebert-base本地存储位置
    embedding_model = "/home/pm/codebert-base"

    # 输入代码，生成嵌入
    code_embedding = generate_code_embeddings(source_code, embedding_model)

    # 存储所有预测结果
    all_predictions = []

    # 设置模态
    configs = [
        {'use_code': True, 'use_tabular': True, 'use_graph': True},  # 全部启用（Baseline）
        {'use_code': False, 'use_tabular': True, 'use_graph': True},  # 去掉 Code 分支
        {'use_code': True, 'use_tabular': False, 'use_graph': True},  # 去掉 Tabular 分支
        {'use_code': True, 'use_tabular': True, 'use_graph': False},  # 去掉 Graph 分支
        {'use_code': True, 'use_tabular': False, 'use_graph': False},  # 只有code 分支
    ]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for config in configs:
    # 构建模型文件的路径
        model_load_path = os.path.join(current_dir, 'model', f"model_{str(config)}.pth")
        # model_load_path = f"model/model_{str(config)}.pth"
        # 预测
        prediction = predict_model(config, model_load_path, code_embedding, device)
        all_predictions.append(prediction)

    # 参数名称
    parameter_names = ["AMAT", "Time", "L1 Hit Rate", "L2 Hit Rate"]

    last_prediction = all_predictions[-1]
    # 将二维张量转换为一维列表
    last_prediction = last_prediction.squeeze().cpu().numpy()  # 使用 squeeze 方法去除多余的维度

    # 构建 prediction_dict
    prediction_dict = {param_name: pred for param_name, pred in zip(parameter_names, last_prediction)}

    # 组合成一个字符串
    result_str = ""
    for param_name in parameter_names:
        result_str += f"{param_name}: {prediction_dict[param_name]:.4f}\n"
    return result_str


if __name__ == "__main__":
    main()