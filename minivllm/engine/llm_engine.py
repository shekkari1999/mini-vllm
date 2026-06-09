import torch

from minivllm.engine.block_manager import BlockAllocator
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import RequestStatus, Sequence
from minivllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model_path, num_blocks, block_size, max_batch_size):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.block_allocator = BlockAllocator(num_blocks, block_size)
        self.scheduler = Scheduler(max_batch_size, self.block_allocator)
        self.model_runner = ModelRunner(model_path, num_blocks, block_size)
        self.seq_counter = 0

    def reset(self):
        """Clear scheduler and block pool between benchmark runs."""
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.seq_counter = 0
        self.block_allocator = BlockAllocator(self.num_blocks, self.block_size)
        self.scheduler.block_allocator = self.block_allocator

    def add_request(self, prompt: str, sampling_params: SamplingParams):
        token_ids = self.model_runner.tokenizer.encode(prompt)
        seq = Sequence(
            request_id=self.seq_counter,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )
        self.seq_counter += 1
        self.scheduler.add_request(seq)

    def step(self):
        seqs = self.scheduler.schedule()
        if not seqs:
            return {}

        prefill_seqs = [seq for seq in seqs if len(seq.output_token_ids) == 0]
        decode_seqs = [seq for seq in seqs if len(seq.output_token_ids) > 0]

        logit_parts = []
        logit_seqs = []
        if prefill_seqs:
            logit_parts.append(self.model_runner.run_prefill(prefill_seqs))
            logit_seqs.extend(prefill_seqs)
        if decode_seqs:
            logit_parts.append(self.model_runner.run_decode(decode_seqs))
            logit_seqs.extend(decode_seqs)

        logits = torch.cat(logit_parts, dim=0)
        eos_id = self.model_runner.tokenizer.eos_token_id

        for seq, logit in zip(logit_seqs, logits):
            next_token = torch.argmax(logit).item()
            seq.output_token_ids.append(next_token)
            if (
                next_token == eos_id
                or len(seq.output_token_ids) >= seq.sampling_params.max_tokens
            ):
                seq.status = RequestStatus.FINISHED

        finished = {}
        for seq in seqs:
            if seq.is_finished():
                for block in seq.block_table:
                    self.block_allocator.free_block(block)
                seq.block_table.clear()
                text = self.model_runner.tokenizer.decode(seq.output_token_ids)
                finished[seq.request_id] = text
        return finished

    def generate(self, prompts, sampling_params):
        for prompt in prompts:
            self.add_request(prompt, sampling_params)
        results = {}
        while self.scheduler.waiting or self.scheduler.running:
            finished = self.step()
            results.update(finished)
        return results
