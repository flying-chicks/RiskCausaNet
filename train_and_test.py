# encoding: utf-8

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from dataloader import random_split, collate
from model import JointReasoning, generate

train_dataset, test_dataset = random_split()
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate)


model = JointReasoning(
    g_in_feat=3072,
    g_n_layers=5,
    g_hidden_size=1024,
    g_out_size=3072,
    n_head=8,
    # llm_path="trained-model"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
num_epochs = 10
tokenizer = AutoTokenizer.from_pretrained("E:/Projects/gemma-7b-it",
                                          )

# 创建一个 SummaryWriter 对象
writer = SummaryWriter('E:/Projects/casual_reasoning/experiment_20240324')

for epoch in range(num_epochs):
    train_losses = []
    for batch_index, batch_data in enumerate(train_dataloader):
        graphs, texts, label01s = batch_data
        token_ids = tokenizer.encode(texts[0], return_tensors='pt', padding=False)

        nce_label = model.calc_nce_label(token_ids, graphs.ndata['embedding'], 0.88)
        _, _, train_loss = model(token_ids, graphs, graphs.ndata['embedding'], nce_label)

        if not train_loss.isnan().item():
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            print(f'epoch: {epoch}, batch: {batch_index}, train loss: {train_loss.item()}')

    train_losses = round(np.mean(train_losses), 6)
    print(f'epoch: {epoch},  train loss: {train_losses}')
    writer.add_scalars('Loss', {'train': train_losses}, epoch)

torch.save(model, 'trained_model/whole_model_parameters.pth')
writer.close()


model.eval()
y_true = []
y_pred = []
for batch_index, batch_data in enumerate(test_dataloader):
    graphs, texts, label01s = batch_data
    parts = texts[0].split("\n")
    label = label01s[0]
    test_text = parts[0] + "\n" + parts[1] + "\n" + parts[2] + "\n"
    try:
        result = generate(tokenizer=tokenizer, model=model, prompt=test_text, graph=graphs,
                          node_feature=graphs.ndata['embedding'], max_length=50, temperature=1.0, top_k=3)
        answer_parts = result.split("\n")[3:]
        answer = "\n".join(answer_parts).lower()
        y_true.append(label)
        if "no" in answer:
            y_pred.append(0)
        elif "yes" in answer:
            y_pred.append(1)
        else:
            y_pred.append(2)
    except Exception as e:
        print(f"An error occurred: {e}")
accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
print(y_true)
print(y_pred)
print(accuracy)
