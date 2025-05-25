import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size: int
        Size of the image (it is a square image).

    patch_size: int
        Size of the patch (it is a square).

    in_chans: int
        Number of input channels (for grey images = 1,
        for colored images = 3).

    embed_dim: int
        The embedding dimension of the transformer.

    Attributes
    ----------
    n_patches: int
        Number of patches inside the image.

    proj: nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int = 3,
        embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (self.img_size // self.patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Run the forward pass.

        Parameters
        ----------
        x: torch.FloatTensor
            of shape `(n_samples, in_chans, img_size, img_size)`

        Returns
        -------
        torch.FloatTensor
            of shape `(n_samples, n_patches, embed_dim)
        """

        # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = self.proj(x)
        x = x.flatten(start_dim=2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """The attention mechanism.

    Parameters
    ----------
    dim: int
        The input and output dimension of per token features.

    n_heads: int
        Number of attention heads

    qkv_bias: bool
        If True then we include bias to the query, key and value
        projections

    attn_p: float
        Dropout probability applied to the query, key and value tensors

    proj_p: float
        Dropout probability applied to the output tensor

    is_causal: bool
        If True, applies attention mask during the forward pass

    Attributes
    ----------
    scale: float
        Normalizing constant for the dot product

    qkv: nn.Linear
        Linear projection for the query, key and value matrices

    proj: nn.Linear
        Linear mapping that takes in the concatenated output of all
        attention heads and maps it into a new space

    attn_drop, proj_drop: nn.Dropout
        The dropout layers
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        qkv_bias: bool = True,
        attn_p: float = 0.,
        proj_p: float = 0.,
        is_causal: bool = False
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.scale = 1 / (self.head_dim ** 0.5)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.is_causal = is_causal

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Run the forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`
            The `+1` corresponds to the [CLS]-Token

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`
            The `+1` corresponds to the [CLS]-Token
        """
        device = x.device
        n_samples, n_tokens, dim = x.size()

        if dim != self.dim:
            raise ValueError(
                f"{dim} of input tensor does not match the"
                f"specified model dimension {self.dim}.")

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = rearrange(qkv, "B T three h d -> three B h T d")

        # each one has shape
        # (n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = torch.einsum("BhQd,BhKd->BhQK", q, k) * self.scale

        if self.is_causal is True:
            mask = torch.tril(torch.ones(n_tokens, n_tokens)).view(
                1, 1, n_tokens, n_tokens
            ).to(device)
            s.masked_fill_(mask == 0, float("-inf"))

        attn = s.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = torch.einsum("BhQK,BhKd->BQhd", attn, v)
        # (n_samples, n_patches + 1, dim)
        weighted_avg = weighted_avg.contiguous().flatten(2)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features: int
        Number of input features

    hidden_features: int
        Number of nodes in the hidden layer

    out_features: int
        Number of output features

    p: float
        Dropout probability

    Attributes
    ----------
    fc1: nn.Linear
        The first linear layer

    act: nn.GeLU
        GeLU activation function

    fc2: nn.Linear
        The second linear layer

    dropout: nn.Dropout
        Dropout applied after second `fc2`
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        p: float = 0.
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Run the forward pass.

        Parameters
        ----------
        x: torch.FloatTensor
            of shape `(n_samples, n_patches + 1, in_features)

        Returns
        -------
        torch.FloatTensor
            of shape `(n_samples, n_patches + 1, out_features)`

        """

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim: int
        Embedding dimension

    n_heads: int
        Number of attention heads

    mlp_ratio: float
        Determines the hidden dimension size of the `MLP` module with
        respect to `dim`

    qkv_bias: bool
        If True, then we include bias to the query, key and value projections.

    p, attn_p: float
        Dropout probability.

    is_causal: bool
        If True, then upper triangular elements in the softmax are masked to 0.

    Attributes
    ----------
    norm1, norm2: nn.LayerNorm
        Layer normalization

    attn: Attention
        Attention module

    mlp: MLP
        MLP module
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        p: float = 0.,
        attn_p: float = 0.,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p, is_causal)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = dim * mlp_ratio
        self.mlp = MLP(dim, hidden_features, dim, p)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Run the forward pass.

        Parameters
        ----------
        x: torch.FloatTensor
            of shape `(n_samples, n_patches + 1, dim)`

        Returns
        -------
        torch.FloatTensor
            of shape `(n_samples, n_patches + 1, dim)`
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Implementation of the Vision transformer.

    Consists only of the transfomer encoder part presented
    in `Attention Is All You Need`-paper.

    Parameters
    ----------
    img_size: int
        Height and width of the image (square image)

    patch_size: int
        Height and width of the patch (square patch)

    in_chans: int
        Number of input_channels

    n_classes: int
        Number of classes

    embed_dim: int
        Dimensionality of the token/patch embeddings

    depth: int
        Number of blocks

    n_heads: int
        Number of attention heads

    mlp_ratio: float
        Determines the hidden dimension of the `MLP` module

    qkv_bias: bool
        If True, then bias is included in query, key and value projections

    p, attn_p: float
        Dropout probability

    is_causal: bool
        If True, then upper triangular elements in the softmax are masked

    Attributes
    ----------
    patch_embed: PatchEmbed
        Instance of `PatchEmbed` layer

    cls_token: nn.Parameter
        Learnable parameter that will represent the first token in the sequence

    pos_emb: nn.Parameter
        Positional embedding of the CLS token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements

    pos_drop: nn.Dropout
        Dropout layer

    blocks: nn.ModuleList[Block]
        List of `Block` modules

    norm: nn.LayerNorm
        Layer normalization
    """

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        n_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = False,
        p: float = 0.,
        attn_p: float = 0.,
        is_causal: bool = False
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                p=p,
                is_causal=is_causal
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Run the forward pass.

        Parameters
        ----------
        x: torch.FloatTensor
            Shape `(n_samples, in_chans, img_size, img_size)`

        Returns
        -------
        logits: torch.FloatTensor
            Logits over all the classes, i.e. shape `(n_classes, n_classes)`
        """

        n_samples = x.size(0)
        x = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # just CLS embedding
        x = self.head(cls_token_final)
        return x
