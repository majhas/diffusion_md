import torch
import torch.nn as nn

from graph_transformer import GraphTransformer

class GTDynamics(nn.Module):
    def __init__(
            self, 
            in_node_nf, 
            in_edge_nf,
            hidden_nf,
            n_heads=8,
            head_nf=64,
            n_processor_layers=1,
            rel_pos_emb=True,
            norm_edges=False,
            with_feedforwards=False,
            gated_residual=True
        ):
        super(GTDynamics, self).__init__()
        
        # plus one to input to account for timestep input
        self.encoder = torch.Linear(in_node_nf+1, hidden_nf, bias=False)
        
        # transform back to positions
        self.decoder = torch.Linear(hidden_nf, in_node_nf, bias=False)

        self.processor = GraphTransformer(
                                    dim=hidden_nf, 
                                    edge_dim=in_edge_nf,
                                    depth=n_processor_layers,
                                    n_heads=n_heads,
                                    dim_head=head_nf,
                                    gated_residual=gated_residual,
                                    rel_pos_emb=rel_pos_emb,
                                    norm_edges=norm_edges,
                                    with_feedforwards=with_feedforwards                                
                                )

    @staticmethod
    def compute_relative_distance(x):
        return torch.cdist(x1=x, x2=x)

    def get_edges(self, x):
        distances = self.compute_relative_distance(x)
        return distances.unsqueeze(-1)

    def forward(self, x, t):
        edges = self.get_edges(x)               # batch_size x n_nodes x n_nodes x 1
        x = torch.concat([x, t], axis=1)        # batch_size x n_nodes x in_node_nf + 1

        x = self.encoder(x)
        x, _ = self.processor(nodes=x, edges=edges)
        return self.decoder(x)
