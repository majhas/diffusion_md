import torch
import torch.nn as nn

from .graph_transformer import GraphTransformer

class GTDynamics(nn.Module):
    def __init__(
            self, 
            n_nodes,
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
        
        self.node_embedding = nn.Embedding(
                                num_embeddings=n_nodes,
                                embedding_dim=in_node_nf
                            )

        # plus one to input to account for timestep input
        self.node_encoder = nn.Linear(in_node_nf+1, hidden_nf, bias=False)

        # predict noise
        self.decoder = nn.Linear(hidden_nf, 3, bias=False)

        self.processor = nn.ModuleList()
        for _ in range(n_processor_layers):

            self.processor.append(GraphTransformer(
                                        dim=hidden_nf, 
                                        edge_dim=in_edge_nf,
                                        depth=n_processor_layers,
                                        heads=n_heads,
                                        dim_head=head_nf,
                                        gated_residual=gated_residual,
                                        rel_pos_emb=rel_pos_emb,
                                        norm_edges=norm_edges,
                                        with_feedforwards=with_feedforwards                                
                                    )
                                )

    @staticmethod
    def compute_relative_distance(x):
        return torch.cdist(x1=x, x2=x)

    def get_edges(self, x):
        distances = self.compute_relative_distance(x)
        return distances.unsqueeze(-1)

    def forward(self, x, pos, t):
        
        t = t.unsqueeze(-1).expand((x.size(0), x.size(1), 1))

        x = self.node_embedding(x).squeeze(-2)

        e = self.get_edges(pos)               # batch_size x n_nodes x n_nodes x 1
        x = torch.concat([x, t], dim=-1)        # batch_size x n_nodes x in_node_nf + 1

        x = self.node_encoder(x)

        for layer in self.processor:
            x, e = layer(nodes=x, edges=e)

        return self.decoder(x)


if __name__ == '__main__':
    ### TEST ###

    dynamics = GTDynamics(n_nodes=5, in_node_nf=64, in_edge_nf=1, hidden_nf=96)

    x = torch.tensor([[0, 1, 2, 3, 4]]).expand((128, 5))
    pos = torch.randn((128, 5, 3))
    t = torch.randint(0, 1001, size=(128, 1)).float()
    t /= 1000

    out = dynamics(x=x, pos=pos, t=t)
    print(out.shape)