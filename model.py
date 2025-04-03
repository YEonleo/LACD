# model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

# We'll import the JSD function from utils:
from utils import jensen_shannon_divergence

class Base_Model:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=24):
        """
        Initializes the model, tokenizer, and related configurations.
        """
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stop_word_ids = []
        self.stopping_criteria = None

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        """
        Loads the tokenizer and model given the model_name.
        Configures device mapping and GPU memory usage if running on CUDA.
        """
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}

        if self.device == "cuda":
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device != "cpu":
            raise ValueError(f"Invalid device: {self.device}")

        # If the model name contains 'vicuna', we assume a different tokenizer path
        tokenizer_name = model_name if 'vicuna' not in model_name else 'huggyllama/llama-7b'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        """
        Sets up stopping criteria for the model based on a list of stop words.
        Each stop word is tokenized and appended to stop_word_ids.
        """
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()

        for stop_word in self.stop_words:
            # Encode "\n<stop_word>" and skip the first 3 tokens
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            self.stop_word_ids.extend(stop_word_ids)
            print(f"Added stop word: {stop_word} with the ids {stop_word_ids}", flush=True)

        stop_words_ids_list = [self.tokenizer.encode('\n' + word)[3:] for word in stop_words]
        self.stopping_criteria.append(LLamaQaStoppingCriteria(stop_words_ids_list))

    def generate(
        self,
        input_text,
        input_text2,
        mode='final_layer_context',
        alpha=0.5,
        layer_alpha=0.1,
        start_layer=17,
        subset_layers=None,
        max_new_tokens=20
    ):
        """
        Generates text based on the specified mode. The advanced modes include:
          - 'final_layer_no_context'
          - 'final_layer_context'
          - 'CAD'
          - 'DOLA'
          - 'contrast_layer_context_nocontext_jsd'
          - 'contrast_layer_context_nocontext_jsd_subset'
        and so on.

        For each mode, calls the appropriate private method. 
        """
        with torch.no_grad():
            # Tokenize input texts
            context_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            no_context_ids = self.tokenizer(input_text2, return_tensors="pt").input_ids.to(self.device)

            initial_length_context = context_ids.shape[1]
            initial_length_no_context = no_context_ids.shape[1]

            # Map mode to methods
            mode_function_map = {
                # Baselines
                'final_layer_no_context': self._generate_final_layer_no_context,
                'final_layer_context': self._generate_final_layer_context,

                # CAD, DOLA
                'CAD': self._generate_CAD,
                'DOLA': self._generate_DOLA,

                # Contrast with JSD
                'contrast_layer_context_nocontext_jsd': self._generate_contrast_layer_adjusted_context_jsd,
                'contrast_layer_context_nocontext_jsd_subset': self._generate_contrast_layer_adjusted_context_jsd_subset
            }

            # Prepare arguments
            mode_args = {
                'final_layer_no_context': (no_context_ids, initial_length_no_context, max_new_tokens),
                'final_layer_context': (context_ids, initial_length_context, max_new_tokens),
                'CAD': (context_ids, no_context_ids, max_new_tokens, initial_length_context, alpha, layer_alpha),
                'DOLA': (context_ids, no_context_ids, max_new_tokens, initial_length_context, alpha, layer_alpha, start_layer),
                'contrast_layer_context_nocontext_jsd': (
                    context_ids, no_context_ids, max_new_tokens,
                    initial_length_context, alpha, layer_alpha, start_layer
                ),
                'contrast_layer_context_nocontext_jsd_subset': (
                    context_ids, no_context_ids, max_new_tokens,
                    initial_length_context, alpha, layer_alpha, subset_layers
                )
            }

            if mode not in mode_function_map:
                valid_modes = list(mode_function_map.keys())
                raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")

            # Call the corresponding method
            return mode_function_map[mode](*mode_args[mode])

    # --------------------------------------------------------------------------------
    # Baseline: final layer only
    # --------------------------------------------------------------------------------

    def _generate_final_layer_context(self, generated_context, initial_length_context, max_new_tokens):
        """
        Baseline - Only the context prompt
        """
        for _ in range(max_new_tokens):
            outputs_context = self.model(generated_context, output_hidden_states=True)
            final_layer_logits_context = self.model.lm_head(outputs_context.hidden_states[-1][:, -1:, :])
            log_probs_final = F.log_softmax(final_layer_logits_context, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)

        new_tokens = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        return new_tokens

    def _generate_final_layer_no_context(self, generated_no_context, initial_length_no_context, max_new_tokens):
        """
        Baseline - Only the no-context prompt
        """
        for _ in range(max_new_tokens):
            outputs_no_context = self.model(generated_no_context, output_hidden_states=True)
            final_layer_logits_no_context = self.model.lm_head(outputs_no_context.hidden_states[-1][:, -1:, :])
            log_probs_final = F.log_softmax(final_layer_logits_no_context, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                break

            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)

        new_tokens = self.tokenizer.decode(
            generated_no_context[0, initial_length_no_context:], skip_special_tokens=True
        )
        return new_tokens

    # --------------------------------------------------------------------------------
    # CAD / DOLA
    # --------------------------------------------------------------------------------

    def _generate_CAD(self, context_ids, no_context_ids,
                      max_new_tokens, initial_length_context, alpha, layer_alpha):
        """
        CAD: (1+alpha)*A_final - alpha*B_final => greedy argmax
        """
        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            outputs_context = self.model(generated_context, output_hidden_states=True)
            outputs_no_context = self.model(generated_no_context, output_hidden_states=True)

            final_logits_context = self.model.lm_head(outputs_context.hidden_states[-1][:, -1:, :])
            final_logits_no_context = self.model.lm_head(outputs_no_context.hidden_states[-1][:, -1:, :])

            adjusted_logits = (1 + alpha)*final_logits_context - alpha*final_logits_no_context
            log_probs_final = F.log_softmax(adjusted_logits, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                break

            # Append the same token to both sequences
            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)

        new_tokens = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        return new_tokens

    def _generate_DOLA(self, context_ids, no_context_ids,
                       max_new_tokens, initial_length_context, alpha, layer_alpha, start_layer):
        """
        DOLA with JSD:
          1. Compute final-layer distribution
          2. Compare final-layer distribution to mid-layer distribution
             across layers >= start_layer, find max JSD
          3. Adjust final distribution with chosen mid-layer
        """
        generated_context = context_ids.clone()

        for _ in range(max_new_tokens):
            outputs_context = self.model(generated_context, output_hidden_states=True)

            # Final layer distribution
            final_logits_context = self.model.lm_head(outputs_context.hidden_states[-1][:, -1:, :])
            probs_final = F.softmax(final_logits_context, dim=-1).squeeze()

            # Compare with each layer >= start_layer
            jsd_divergences = []
            mid_probs_list = []

            for layer_idx, hidden_state in enumerate(outputs_context.hidden_states):
                if layer_idx >= start_layer:
                    mid_logits_context = self.model.lm_head(hidden_state[:, -1:, :])
                    mid_probs = F.softmax(mid_logits_context, dim=-1).squeeze()

                    jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                    jsd_divergences.append((jsd_val.item(), layer_idx))
                    mid_probs_list.append(mid_probs)

            # If no valid layers found, fallback to final
            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
            else:
                # Pick layer with maximum JSD
                _, max_jsd_index = max(jsd_divergences, key=lambda x: x[0])
                offset_idx = max_jsd_index - start_layer
                if offset_idx < 0:
                    offset_idx = 0
                chosen_mid_probs = mid_probs_list[offset_idx]

                if layer_alpha == 0:
                    idea_probs = probs_final - chosen_mid_probs
                else:
                    idea_probs = (1 + layer_alpha)*probs_final - layer_alpha*chosen_mid_probs

                next_token_id = torch.argmax(idea_probs).unsqueeze(0)

            if next_token_id.item() in self.stop_word_ids:
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)

        # Return newly generated tokens
        new_tokens = self.tokenizer.decode(generated_context[0, initial_length_context:], skip_special_tokens=True)
        return new_tokens

    # --------------------------------------------------------------------------------
    # (A-B vs A-B) JSD
    # --------------------------------------------------------------------------------

    def _generate_contrast_layer_adjusted_context_jsd(
        self,
        context_ids,
        no_context_ids,
        max_new_tokens,
        initial_length_context,
        alpha,
        layer_alpha,
        start_layer
    ):
        """
        Compares final-layer (A-B) to mid-layer (A-B) for each layer >= start_layer,
        uses JSD to choose the best mid-layer, then adjusts the final distribution.
        """
        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            out_ctx = self.model(generated_context, output_hidden_states=True)
            out_nocxt = self.model(generated_no_context, output_hidden_states=True)

            final_logits_context = self.model.lm_head(out_ctx.hidden_states[-1][:, -1:, :])
            final_logits_no_context = self.model.lm_head(out_nocxt.hidden_states[-1][:, -1:, :])
            final_adjusted_logits = (1 + alpha)*final_logits_context - alpha*final_logits_no_context
            probs_final = F.softmax(final_adjusted_logits, dim=-1).squeeze()

            jsd_divergences = []
            mid_probs_list = []

            for layer_idx, (ctx_hid, nocxt_hid) in enumerate(
                zip(out_ctx.hidden_states, out_nocxt.hidden_states)
            ):
                if layer_idx >= start_layer:
                    mid_logits_ctx = self.model.lm_head(ctx_hid[:, -1:, :])
                    mid_logits_nocxt = self.model.lm_head(nocxt_hid[:, -1:, :])
                    mid_adjusted_logits = (1 + alpha)*mid_logits_ctx - alpha*mid_logits_nocxt
                    mid_probs = F.softmax(mid_adjusted_logits, dim=-1).squeeze()

                    jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                    jsd_divergences.append((jsd_val.item(), layer_idx))
                    mid_probs_list.append(mid_probs)

            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
            else:
                _, max_jsd_idx = max(jsd_divergences, key=lambda x: x[0])
                offset_idx = max_jsd_idx - start_layer
                if offset_idx < 0:
                    offset_idx = 0
                chosen_mid_probs = mid_probs_list[offset_idx]

                if layer_alpha == 0:
                    idea_probs = probs_final - chosen_mid_probs
                else:
                    idea_probs = (1 + layer_alpha)*probs_final - layer_alpha*chosen_mid_probs

                next_token_id = torch.argmax(idea_probs).unsqueeze(0)

            if next_token_id.item() in self.stop_word_ids:
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)

        output_text = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        return output_text

    def _generate_contrast_layer_adjusted_context_jsd_subset(
        self,
        context_ids,
        no_context_ids,
        max_new_tokens,
        initial_length_context,
        alpha,
        layer_alpha,
        subset_layers
    ):
        """
        Similar to the above, but only checks the specified subset of layers instead of all layers >= start_layer.
        """
        if subset_layers is None:
            subset_layers = []

        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            out_ctx = self.model(generated_context, output_hidden_states=True)
            out_nocxt = self.model(generated_no_context, output_hidden_states=True)

            # Final distribution (A-B)
            final_logits_ctx = self.model.lm_head(out_ctx.hidden_states[-1][:, -1:, :])
            final_logits_nocxt = self.model.lm_head(out_nocxt.hidden_states[-1][:, -1:, :])
            final_adjusted_logits = (1 + alpha)*final_logits_ctx - alpha*final_logits_nocxt
            probs_final = F.softmax(final_adjusted_logits, dim=-1).squeeze()

            jsd_divergences = []
            mid_probs_list = []

            for layer_idx in subset_layers:
                if 0 <= layer_idx < len(out_ctx.hidden_states):
                    mid_logits_ctx = self.model.lm_head(out_ctx.hidden_states[layer_idx][:, -1:, :])
                    mid_logits_nocxt = self.model.lm_head(out_nocxt.hidden_states[layer_idx][:, -1:, :])
                    mid_adjusted_logits = (1 + alpha)*mid_logits_ctx - alpha*mid_logits_nocxt
                    mid_probs = F.softmax(mid_adjusted_logits, dim=-1).squeeze()

                    jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                    jsd_divergences.append((jsd_val.item(), layer_idx))
                    mid_probs_list.append(mid_probs)

            # If subset_layers is empty or invalid, fallback
            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
            else:
                _, max_jsd_idx = max(jsd_divergences, key=lambda x: x[0])

                # find the correct offset in mid_probs_list
                chosen_mid_probs = None
                for i, (val, lidx) in enumerate(jsd_divergences):
                    if lidx == max_jsd_idx:
                        chosen_mid_probs = mid_probs_list[i]
                        break

                if chosen_mid_probs is None:
                    next_token_id = torch.argmax(probs_final).unsqueeze(0)
                else:
                    if layer_alpha == 0:
                        idea_probs = probs_final - chosen_mid_probs
                    else:
                        idea_probs = (1 + layer_alpha)*probs_final - layer_alpha * chosen_mid_probs

                    # Optionally re-softmax
                    idea_probs = F.softmax(idea_probs, dim=-1)
                    next_token_id = torch.argmax(idea_probs).unsqueeze(0)

            if next_token_id.item() in self.stop_word_ids:
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)

        output_text = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        return output_text
