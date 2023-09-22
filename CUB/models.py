
from CUB.template_model import MLP, inception_v3, End2EndModel


# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class, residue, should_detach):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                        three_class=three_class, residue = residue, should_detach = should_detach)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim, residue):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes + residue, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid, residue, residue_indept, should_detach, disentangle, mi_args):
    model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3), residue = residue, should_detach = should_detach)
    if n_class_attr == 3:
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes + residue + residue_indept, num_classes=num_classes, expand_dim=expand_dim)

    if residue_indept > 0:
        model3 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=residue_indept, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3), residue_indept = residue_indept)

        return End2EndModel(model1, model2, model3, n_attributes, residue, use_relu, use_sigmoid, n_class_attr, should_detach, disentangle, mi_args)
    else:
        return End2EndModel(model1, model2, None, n_attributes, residue, use_relu, use_sigmoid, n_class_attr, should_detach, disentangle, mi_args)

# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux)

# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, use_aux, n_attributes, three_class, connect_CY):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=False, three_class=three_class,
                        connect_CY=connect_CY)
