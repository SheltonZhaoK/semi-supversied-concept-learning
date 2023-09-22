import torch

def calculate_ortho_loss(concept_fn_out, residual_fn=None):
    cfn_out = concept_fn_out.reshape(-1, concept_fn_out.shape[-1])
    if residual_fn is not None:
        residual_fn = residual_fn.reshape(-1, residual_fn.shape[-1])
    cfn_out = cfn_out - torch.mean(cfn_out, dim=0)
    if residual_fn == None:
        residual_fn = cfn_out
        was_none = True
    else:
        was_none = False
    residual_fn = residual_fn - torch.mean(residual_fn, dim=0)
    ortho_matrix = torch.matmul(
        torch.transpose(cfn_out, 0, 1),
        residual_fn,
    )
    ortho_matrix = torch.div(ortho_matrix, cfn_out.shape[0])
    concept_std = torch.std(cfn_out, dim=0, unbiased=False)
    residual_std = torch.std(residual_fn, dim=0, unbiased=False)
    division = torch.ger(concept_std, residual_std)
    ortho_matrix = torch.div(ortho_matrix, division)
    if was_none:
        ortho_matrix = ortho_matrix[
            ~torch.eye(ortho_matrix.shape[0], ortho_matrix.shape[1], dtype=bool)
        ]
    ortho_loss = torch.mean(torch.abs(ortho_matrix))
    return ortho_loss