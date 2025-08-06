import torch
from models.TimeCMA import Dual  # 假设你的代码保存在 model.py 中

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_nodes = 72
    seq_len = 120
    pred_len = 120
    d_llm = 72

    # 初始化模型
    model = Dual(
        device=device,
        channel=32,
        num_nodes=num_nodes,
        seq_len=seq_len,
        pred_len=pred_len,
        dropout_n=0.1,
        d_llm=d_llm,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8
    ).to(device)

    print(f"Total Parameters: {model.count_trainable_params()}")

    # 构造模拟输入
    input_data = torch.randn(batch_size, seq_len, num_nodes).to(device)
    input_data_mark = torch.randn(batch_size, seq_len, num_nodes).to(device)  # 假设需要同样 shape
    embeddings = torch.randn(batch_size, d_llm, num_nodes, 1).to(device)      # [B, E, N, 1]

    # 前向传播
    with torch.no_grad():
        output = model(input_data, input_data_mark, embeddings)

    print(f"Output shape: {output.shape}")  # 应为 [B, L, N]，即 [4, 96, 7]

if __name__ == '__main__':
    main()
