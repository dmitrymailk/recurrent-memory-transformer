import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import transformers
import gc
from transformers.models.gpt2 import GPT2LMHeadModel
from cut_cross_entropy.transformers.llama import (
    cce_forward,
    linear_cross_entropy,
    _PATCH_OPTS,
)


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens  # 16
        embeddings = self.model.get_input_embeddings()  # torch.Size([50257, 768])
        memory_dim = getattr(
            self.model.config, "n_embd", self.model.config.hidden_size
        )  # 768
        memory_weights = (
            torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        )  # torch.Size([16, 768])
        self.register_parameter(
            "memory", torch.nn.Parameter(memory_weights, requires_grad=True)
        )

        self.read_memory_position = range(num_mem_tokens)  # range(0, 16)
        self.write_memory_position = range(-num_mem_tokens, 0)  # range(-16, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:  # first segment=None, second segment=
            memory_state = self.set_memory(input_ids.shape)
        # memory_state=torch.Size([4, 16, 768])
        seg_kwargs = self.process_input(
            input_ids, memory_state, write_mem=True, **kwargs
        )  # inputs_embeds=torch.Size([4, 544, 768]),
        # print(seg_kwargs["inputs_embeds"].shape)
        # seg_kwargs['labels_mask'] = kwargs['labels_mask']
        out = self.model(**seg_kwargs)  # out.logits=torch.Size([4, 544, 50257])
        out, new_memory_state = self.process_output(out, **kwargs)
        # out.logits=torch.Size([4, 512, 50257])
        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids, memory_state, attention_mask=attention_mask, write_mem=False
        )
        out = self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs["attention_mask"],
            **generate_kwargs,
        )
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)
        # firts segment, input_ids=torch.Size([4, 512])
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:  # fist segment=None, second segment=
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if self.num_mem_tokens > 0:  # True, 16
            if write_mem:  # True, training
                inputs_embeds = torch.cat(
                    [
                        memory_state,
                        inputs_embeds,
                        memory_state,
                    ],
                    dim=1,
                )
            else:
                inputs_embeds = torch.cat(
                    [
                        memory_state,
                        inputs_embeds,
                    ],
                    dim=1,
                )

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds  # torch.Size([4, 544, 768])
        if kwargs.get("attention_mask") is not None:  # True, torch.Size([4, 512])
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"], inputs_embeds.shape
            )  # torch.Size([4, 544])
        # if kwargs.get("labels_mask") is not None:  # True, torch.Size([4, 512])
        #     seg_kwargs["labels_mask"] = self.pad_labels_mask(
        #         kwargs["labels_mask"], inputs_embeds.shape
        #     )  # torch.Size([4, 544])
        # seg_kwargs["output_hidden_states"] = True
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[
                :, self.num_mem_tokens : self.num_mem_tokens + attention_mask.shape[1]
            ] = attention_mask
            return mask

    def pad_labels_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.zeros(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[
                :, self.num_mem_tokens : self.num_mem_tokens + attention_mask.shape[1]
            ] = attention_mask
            return mask.bool()

    def _process_output(self, model_outputs, **kwargs):
        """ORIGINAL"""
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            out["logits"] = model_outputs.logits[
                :, self.num_mem_tokens : -self.num_mem_tokens
            ]

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                    for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state

    def process_output(self, model_outputs, **kwargs):
        """new version"""
        if self.num_mem_tokens not in {0, None}:  # True
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            # out["logits"] = model_outputs.logits[
            #     :, self.num_mem_tokens : -self.num_mem_tokens
            # ]
            out["logits"] = model_outputs.logits

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                    for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


import random


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs  # {'segment_size': 512, 'max_n_segments': 2, 'segment_alignment': None, 'k2': -1}

    def forward(
        self,
        input_ids,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        memory_state = None  # input_ids=torch.Size([4, 1017]), inputs_embeds=None
        segmented = self.segment(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels_mask=labels_mask,
        )  # len(segmented)=2, segmented[0]['input_ids']=torch.Size([4, 512])
        # inputs_embeds = None # не помогает оптимизировать память
        # gc.collect()
        # segmented[1]['input_ids'].shape=torch.Size([4, 505])
        cell_outputs = []
        # print('\n\n\nForward: ', [s['input_ids'].shape for s in segmented])
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(
                **segment,
                memory_state=memory_state,
                output_hidden_states=True,
                # labels_mask=labels_mask,
            )
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(
            cell_outputs,
            labels=labels,
            labels_mask=labels_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        # print('\n\n\nGenerate: ', [s['input_ids'].shape for s in segmented])
        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(
                **segment, memory_state=memory_state, output_hidden_states=True
            )

        final_segment = segmented[-1]
        out = self.memory_cell.generate(
            **final_segment, memory_state=memory_state, **generate_kwargs
        )

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get("segment_alignment")  # None
        segment_size = self.rmt_config.get("segment_size")  # 512
        if align in {"left", None}:  # True
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [
                tensor.shape[1]
            ]  # [0, 512, 1017]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align in {"right", None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align == "center":
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments  # [torch.Size([4, 512]), torch.Size([4, 505]), torch.Size([4, 512])]

    def _process_outputs(self, cell_outputs, **kwargs):
        """ORIGINAL"""
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple(
            [
                torch.cat(layer_hs, dim=1)
                for layer_hs in zip(*[o.hidden_states for o in cell_outputs])
            ]
        )

        labels = kwargs.get("labels")
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get("labels_mask")
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out["loss"] = loss_fct(flat_logits, flat_labels)
            if out["loss"] is None:
                raise ValueError
        else:
            out["loss"] = 0

        out["logits"] = full_logits
        segment_keys = ["loss", "logits"]
        if kwargs.get("output_attentions"):
            segment_keys.append("attentions")
        if kwargs.get("output_hidden_states"):
            segment_keys.append("hidden_states")
            out["hidden_states"] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f"{key}_{seg_num}"] = value

        return out

    def process_outputs(self, cell_outputs, **kwargs):
        """new version"""
        out = CausalLMOutputWithCrossAttentions()
        # full_logits = torch.cat(
        #     [o.logits for o in cell_outputs if not o.logits is None],
        #     dim=0,
        # )
        # full_hidden_states = tuple(
        #     [
        #         torch.cat(layer_hs, dim=1)
        #         for layer_hs in zip(*[o.hidden_states for o in cell_outputs])
        #     ]
        # )
        total_hidden_states = torch.cat(
            [item.hidden_states[-1] for item in cell_outputs],
            dim=1,
        )

        labels = kwargs.get("labels")
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = total_hidden_states[
                ..., :-1, :
            ].contiguous()  # full_logits=torch.Size([4, 1024, 50257])
            # shift_logits = (
            #     full_logits.contiguous()
            # )  # full_logits=torch.Size([4, 1024, 50257])
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get("labels_mask")
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]  # torch.Size([15])
                flat_logits = flat_logits[
                    shift_mask.view(-1)
                ]  # torch.Size([15, 50257])
            # print(flat_logits.shape)
            # tensor([37648,  3823, 50256, 50256, 36269, 50256, 50256, 15813,  6607, 50256, 50256, 37648,  3823, 50256, 50256], device='cuda:0')
            # loss= tensor(13.1875, device='cuda:0', dtype=torch.bfloat16, grad_fn=<NllLossBackward0>)
            # self.memory_cell.model.lm_head
            # flat_logits = self.memory_cell.model.lm_head(flat_logits)
            loss = linear_cross_entropy(
                flat_logits,
                self.memory_cell.model.lm_head.weight,
                flat_labels,
                shift=False,
                impl=_PATCH_OPTS.impl,
                reduction=_PATCH_OPTS.reduction,
            )
            # out["loss"] = loss_fct(flat_logits, flat_labels)
            out["loss"] = loss
            if out["loss"] is None:
                raise ValueError
        else:
            out["loss"] = 0

        # out["logits"] = full_logits
        out["logits"] = flat_logits
        segment_keys = ["loss", "logits"]
        if kwargs.get("output_attentions"):
            segment_keys.append("attentions")
        # if kwargs.get("output_hidden_states"):
        #     segment_keys.append("hidden_states")
        #     out["hidden_states"] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f"{key}_{seg_num}"] = value

        return out

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get("k2"), self.rmt_config.get(
            "max_n_segments"
        )
        if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
            return memory_state

        memory_state = memory_state.detach()
        return memory_state
