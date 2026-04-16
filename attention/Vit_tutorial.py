import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

### Step 1 — Patch Embeddin
class PatchEmbedding(nn.Module):
    # image dataset size = B, 3, 224, 244 
    # number of patches = (224/16) * (224/16) = 14 * 14 = 196
    # transformer input = B, 197, 768 (if embedding dimension is 768)

    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)           # (B, embed_dim, 14, 14)
        x = x.flatten(2)           # (B, embed_dim, 196)
        x = x.transpose(1, 2)      # (B, 196, embed_dim)
        return x

### Step 2 — Multi-Head Self Attention
class Attention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        
        qkv = self.qkv(x)  
        qkv = qkv.reshape(B, N, 3, self.n_heads, D // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.proj(out)

### Step 3 — Transformer Encoder Block
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
 
### Step 4 — Full Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_ch=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        n_heads=12
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_ch, embed_dim
        )
        
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)  # (B, N, D)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_output = x[:, 0]
        return self.head(cls_output)
    
if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    options.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    options.add_argument("--data_dir", type=str, default="./data", help="Directory for storing the dataset")
    options.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    options.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    options.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    options.add_argument("--patch_size", type=int, default=16, help="Size of each patch")
    args = options.parse_args()


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/home/basheer/Signapse/Codes/DeepLearning-Performance-Lab/data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = VisionTransformer(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(args.num_epochs):
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss {loss.item():.4f}")
    


