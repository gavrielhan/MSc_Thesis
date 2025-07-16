import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv, GCNConv, Linear
import torch.nn.functional as F
from link_prediction_head import CoxHead, TimeAwareCosineLinkPredictor

# Model and head definitions
class HeteroGAT(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim: int = 64,
        out_dim:   int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout:   float = 0.2
    ):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = nn.ModuleDict({
            n: Linear(in_dims[n], hidden_dim) for n in metadata[0]
        })
        self.shrink = nn.ModuleDict({
            n: nn.Identity() for n in metadata[0]
        })

        def make_layer(in_d, out_d):
            convs = {}
            for et in metadata[1]:
                if et in [
                    ('patient','to','signature'),
                    ('signature','to_rev','patient')
                ]:
                    convs[et] = GATConv(
                        (in_d, in_d),
                        out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        edge_dim=1,
                    )
                elif et in [
                    ('patient','follows','patient'),
                    ('patient','follows_rev','patient')
                ]:
                    convs[et] = GCNConv(in_d, out_d)
                else:
                    convs[et] = GATConv(
                        (in_d, in_d),
                        out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
            return HeteroConv(convs, aggr='sum')

        # Build a list of layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i > 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else out_dim
            self.convs.append(make_layer(in_d, out_d))

        self.linear_proj = nn.ModuleDict({
            n: Linear(out_dim, out_dim) for n in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = {n: self.input_proj[n](x) for n, x in x_dict.items()}
        ew = {}
        fwd = ('patient','to','signature')
        rev = ('signature','to_rev','patient')
        if edge_attr_dict and fwd in edge_attr_dict:
            w = edge_attr_dict[fwd].view(-1,1).float()
            ew[fwd] = w
            ew[rev] = w.flip(0)

        for conv in self.convs:
            x_raw = conv(x, edge_index_dict, edge_attr_dict=ew)
            x = {n: F.relu(x_raw.get(n, x[n]) + x[n]) for n in self.metadata[0]}

        out = {n: self.linear_proj[n](h) for n, h in x.items()}
        return out

class JointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_link = nn.Parameter(torch.tensor(0.0))
        self.s_cox = nn.Parameter(torch.tensor(0.0))
    def forward(self, link_loss: torch.Tensor, cox_loss: torch.Tensor) -> torch.Tensor:
        log_var_link = F.softplus(self.s_link)
        log_var_cox = F.softplus(self.s_cox)
        term_link = 0.5 * link_loss * torch.exp(-log_var_link) + 0.5 * log_var_link
        term_cox = 0.5 * cox_loss * torch.exp(-log_var_cox) + 0.5 * log_var_cox
        return term_link + term_cox

# Model and head instantiations
in_dims = {
    'patient': 138,
    'signature': 96,
    'condition': 7,
}
metadata = (
    ['patient','condition','signature'],
    [
      ('patient','to','signature'),
      ('signature','to_rev','patient'),
      ('patient','has','condition'),
      ('condition','has_rev','patient'),
      ('patient','follows','patient'),
      ('patient','follows_rev','patient'),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeteroGAT(
    metadata=metadata,
    in_dims=in_dims,
    hidden_dim=64,
    out_dim=64,
    dropout=0.2,
    num_heads=4,      # Increased attention heads
    num_layers=3      # More GAT layers
).to(device)
link_head = TimeAwareCosineLinkPredictor(init_scale=5.0, use_bias=True).to(device)
cox_head = CoxHead(input_dim=64, num_conditions=7).to(device)
joint_head = JointHead().to(device)
patient_classifier = nn.Linear(64, 7).to(device)