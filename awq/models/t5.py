from .base import BaseAWQForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTDecoderLayer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Block
from transformers import AutoModelForSeq2SeqLM
def sig(x):
    import inspect
    print(inspect.signature(x))

class FlanT5AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = None
    HF_AUTO_CLASS = AutoModelForSeq2SeqLM

    @staticmethod
    def get_model_layers(model: T5ForConditionalGeneration):
        # layers = []
        # layers.extend(model.encoder.block)
        # layers.extend(model.decoder.block)

        return model.encoder.block
    
    @staticmethod
    def get_act_for_scaling(module: T5Block):
        # I think that T5s and FlanT5s will need to be treated differently
        # Guess: 
        # T5s will need a scale_layer in the MLP block
        # FlanT5s will not need a scale layer (gated MLP like llama?)
        # We can look for this in the module config - one of the entries controls this.
        # is_gated_activation
        # ////
        # We might want to use this to scale V & K in the cross attn block
        return dict(
            is_scalable=False 
        )
    
    @staticmethod
    def move_embed(model: T5ForConditionalGeneration, device: str):
        model.shared = model.shared.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: T5Block, input_feat, module_kwargs):
        layers = []
        # self attention input
        # layer 0 is always self attn
        layers.append(dict(
                prev_op=module.layer[0].layer_norm,
                layers = [
                    module.layer[0].SelfAttention.q,
                    module.layer[0].SelfAttention.k,
                    module.layer[0].SelfAttention.v
                ],
                inp=input_feat['layer.0.SelfAttention.q'],
                module2inspect=module.layer[0].SelfAttention,
                kwargs={
                    "mask": module_kwargs["attention_mask"],   
                }
        ))
        

        layers.append(dict(
            prev_op=module.layer[0].SelfAttention.v,
            layers=[
                module.layer[0].SelfAttention.o
            ],
            inp=input_feat['layer.0.SelfAttention.o'],
            # kwargs={
            #         "attention_mask": module_kwargs["attention_mask"],   
            #     }

        ))

        if module.is_decoder: # decoder branch w/ cross attn
            # Why doesnt this work:
            # stupid huggingface t5 implementation 
            # the code to capture the activations doesnt include encoder hidden states so this block is skipped
            # We are going to NOT scale these entries and see what happens...

            # # import pdb; pdb.set_trace()
            # layers.append(dict(
            #     prev_op=module.layer[1].layer_norm,
            #     layers = [
            #         module.layer[1].EncDecAttention.q,
            #         module.layer[1].EncDecAttention.k,
            #         module.layer[1].EncDecAttention.v
            #     ],
            #     inp=input_feat['layer.1.EncDecAttention.q'],
            #     module2inspect=module.layer[1].EncDecAttention,
            #     kwargs=module_kwargs
            # ))

            # layers.append(dict(
            #     prev_op=module.layer[1].EncDecAttention.v,
            #     layers=[module.layer[1].EncDecAttention.o],
            #     inp=input_feat['layer.1.EncDecAttention.o'],
            #     module2inspect=module.layer[1].EncDecAttention,
            # ))

            # dense relu dense 
            layers.append(dict(
                prev_op=module.layer[2].layer_norm,
                layers=[
                    module.layer[2].DenseReluDense.wi_0,
                    module.layer[2].DenseReluDense.wi_1
                    ],
                inp=input_feat['layer.2.DenseReluDense.wi_0'],
                module2inspect=module.layer[2].DenseReluDense,
            ))

        else: # encoder branch w/o cross attn
            # dense relu dense
            layers.append(dict(
                prev_op=module.layer[1].layer_norm,
                layers=[
                    module.layer[1].DenseReluDense.wi_0,
                    module.layer[1].DenseReluDense.wi_1
                    ],
                inp=input_feat['layer.1.DenseReluDense.wi_0'],
                module2inspect=module.layer[1].DenseReluDense,
            ))

        return layers