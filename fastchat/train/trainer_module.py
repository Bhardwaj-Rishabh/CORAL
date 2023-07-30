from transformers import Trainer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import sys

class CustomTrainerFT(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss



#-------------------------for pos neg training-setup-1------------------------
def compute_contrastive_loss(sample_type, logits, labels, vocab_size):
    '''
    expecting number of elements to be divisbile by 3
    '''
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    #
    loss = 0.
    for l in range(shift_labels.shape[0]):
        #2,4 signifies pos and neutral sample
        if sample_type[l] in [2,4]:
            loss_fct = CrossEntropyLoss()
            nl = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
            loss+=nl
        elif sample_type[l]==3:
            #3 signifies neg sample
            loss_fct = CrossEntropyLoss()
            nl = -loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
            loss+=nl
        else:
            breakpoint()
            print(">>>>Error: Not sure if it is positive sample or negative")
            sys.exit()
    return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #correcting the first token
        sample_type = torch.clone(inputs['input_ids'][:,0])
        inputs['input_ids'][:,0] = 1
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #loss = outputs["loss"]['sum_loss']
        loss = compute_contrastive_loss(sample_type, outputs['logits'], outputs['loss']['labels'], outputs['loss']['vocab_size'])
        #print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss


#-------------------------for pos neg training-setup-2------------------------
def compute_contrastive_loss_setup2(sample_type, logits, labels, vocab_size):
    '''
    expecting number of elements to be divisbile by 3
    '''
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    #
    loss = 0.
    count_invals = 0
    for l in range(shift_labels.shape[0]):

        #2,4 signifies pos and neutral sample
        if sample_type[l] in [2,4]:
            loss_fct = CrossEntropyLoss()
            nl = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
                count_invals += 1

            loss+=nl

        elif sample_type[l]==3:
            #3 signifies neg sample
            loss_fct = CrossEntropyLoss()
            nl_neg = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            if nl_neg > 1:
                nl = torch.tensor(0.0).to(shift_labels.device)
            else:
                nl = -0.1*nl_neg
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl_neg = torch.tensor(0.0).to(shift_labels.device)
                nl = nl_neg
                count_invals += 1

            loss+=nl

        else:
            breakpoint()
            print(">>>>Error: Not sure if it is positive sample or negative")
            sys.exit()
            
    if loss == 0:
        loss = nl_neg*1e-5
            
    print(f"count NANs: {count_invals}")
    print(f"div factor: {shift_labels.shape[0]-count_invals}")
    loss = loss/(shift_labels.shape[0]-count_invals)
    print(f"\t\t\tOverall loss={loss}")
    return loss

class CustomTrainer_setup2(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #correcting the first token
        sample_type = torch.clone(inputs['input_ids'][:,0])
        inputs['input_ids'][:,0] = 1
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #loss = outputs["loss"]['sum_loss']
        loss = compute_contrastive_loss_setup2(sample_type, outputs['logits'], outputs['loss']['labels'], outputs['loss']['vocab_size'])
        #print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss


#-------------------------for pos neg training-setup-3------------------------
def compute_contrastive_loss_setup3(sample_type, logits, labels, vocab_size):
    '''
    expecting number of elements to be divisbile by 3
    '''
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    #
    loss = 0.
    count_invals = 0
    for l in range(shift_labels.shape[0]):

        #2,4 signifies pos and neutral sample
        if sample_type[l] in [2,4]:
            loss_fct = CrossEntropyLoss()
            nl = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
                count_invals += 1

            loss+=nl

        elif sample_type[l]==3:
            #3 signifies neg sample
            loss_fct = CrossEntropyLoss()
            nl_neg = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            if nl_neg > 1:
                nl = nl_neg
                print("--------------neg sample loss is minimized----------------")
            else:
                nl = -0.1*nl_neg
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl_neg = torch.tensor(0.0).to(shift_labels.device)
                nl = nl_neg
                count_invals += 1

            loss+=nl

        else:
            breakpoint()
            print(">>>>Error: Not sure if it is positive sample or negative")
            sys.exit()
            
    print(f"count NANs: {count_invals}")
    print(f"div factor: {shift_labels.shape[0]-count_invals}")
    loss = loss/(shift_labels.shape[0]-count_invals)
    print(f"\t\t\tOverall loss={loss}")
    return loss

class CustomTrainer_setup3(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #correcting the first token
        sample_type = torch.clone(inputs['input_ids'][:,0])
        inputs['input_ids'][:,0] = 1
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #loss = outputs["loss"]['sum_loss']
        loss = compute_contrastive_loss_setup3(sample_type, outputs['logits'], outputs['loss']['labels'], outputs['loss']['vocab_size'])
        #print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss




#-------------------------for pos neg training-setup-4------------------------
def compute_contrastive_loss_setup4(sample_type, logits, labels, vocab_size):
    '''
    expecting number of elements to be divisbile by 3
    '''
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    #
    loss = 0.
    count_invals = 0
    for l in range(shift_labels.shape[0]):

        #2,4 signifies pos and neutral sample
        if sample_type[l] in [2,4]:
            loss_fct = CrossEntropyLoss()
            nl = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
                count_invals += 1

            loss+=nl

        elif sample_type[l]==3:
            #3 signifies neg sample
            loss_fct = CrossEntropyLoss()
            nl_neg = loss_fct(shift_logits[l,:,:], shift_labels[l,:])
            if nl_neg > 1:
                nl = nl_neg
                print("--------------neg sample loss is minimized----------------")
            else:
                nl = -0.1*nl_neg
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                #print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl_neg = torch.tensor(0.0).to(shift_labels.device)
                nl = nl_neg
                count_invals += 1

            loss+=nl

        else:
            breakpoint()
            print(">>>>Error: Not sure if it is positive sample or negative")
            sys.exit()
            
    print(f"count NANs: {count_invals}")
    print(f"div factor: {shift_labels.shape[0]-count_invals}")
    loss = loss/(shift_labels.shape[0]-count_invals)
    print(f"\t\t\tOverall loss={loss}")
    return loss

class CustomTrainer_setup4(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #correcting the first token
        sample_type = torch.clone(inputs['input_ids'][:,0])
        inputs['input_ids'][:,0] = 1
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #loss = outputs["loss"]['sum_loss']
        loss = compute_contrastive_loss_setup4(sample_type, outputs['logits'], outputs['loss']['labels'], outputs['loss']['vocab_size'])
        #print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss




#LLaMA custom loss
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class CustomLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss={'sum_loss':loss, 'logits': logits, 'labels': labels, 'vocab_size': self.config.vocab_size},
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
