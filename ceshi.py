from model import build
a = build('vitb32')
for name, value in a.named_parameters():
    if 'transformer' in name and 'visual' not in name:
        print(name)
    if name == 'token_embedding.weight' or name == 'ln_final.weight' or name == 'ln_final.bias':
        print(name)