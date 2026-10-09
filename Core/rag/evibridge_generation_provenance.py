"""Record client-side generation attempts without retaining prompts or responses."""
import copy
import hashlib


def text_sha256(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class GenerationRecorder:
    def __init__(self):
        self.calls = []
        self.context = []
        self.retained_call_index = None

    def call(self, llm, prompt, evidence, stage):
        context = copy.deepcopy(evidence)
        for positional in (False, True):
            record = {
                "stage": stage,
                "call_style": "positional" if positional else "keyword",
                "prompt_sha256": text_sha256(prompt),
                "prompt_block_ids": [int(x["block_id"]) for x in context if x.get("block_id") is not None],
                "block_text_sha256": [{"block_id": int(x["block_id"]), "sha256": text_sha256(x.get("text", ""))}
                                      for x in context if x.get("block_id") is not None],
                "success": False,
                "response_sha256": None,
                "error_type": None,
            }
            self.calls.append(record)
            try:
                if positional:
                    response = llm.get_completion(prompt)
                else:
                    response = llm.get_completion(prompt=prompt, json_response=False)
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                if isinstance(exc, TypeError) and not positional:
                    continue
                raise
            record["success"] = True
            record["response_sha256"] = text_sha256(response)
            self.context = context
            self.retained_call_index = len(self.calls) - 1
            return response

    def snapshot(self):
        active = self.calls[self.retained_call_index] if self.retained_call_index is not None else None
        return {
            "schema_version": 1,
            "recording_scope": "client_call_attempts",
            "calls": copy.deepcopy(self.calls),
            "retained_call_index": self.retained_call_index,
            "retained_stage": active["stage"] if active else None,
            "prompt_block_ids": list(active["prompt_block_ids"]) if active else None,
            "prompt_sha256": active["prompt_sha256"] if active else None,
        }