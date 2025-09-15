from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import json
from typing import List, Optional


class LocalLLM:
    """A class to manage a GGUF model for CPU inference using llama-cpp-python."""
    def __init__(self, model_id: str, model_file: str, n_ctx: int = 4096, chat_format: str = "chatml", flash_attn: bool = True, verbose: bool = True):
        print(f"--- Initializing GGUF LLM for CPU: {model_id} ---")
        model_path = hf_hub_download(repo_id=model_id, filename=model_file)

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            chat_format=chat_format,
            flash_attn=flash_attn,
            n_threads=-1,
            verbose=verbose)

        print("--- LLM Initialized and Patched Successfully ---")

    def decompose_query(self, prompt: str, user_query: str) -> list:
        """Generates a list of subqueries using the model's JSON mode."""
        try:
            response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                repeat_penalty=1.3,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["queries"]
                    }
                }
            )
            content = response['choices'][0]['message']['content']
            parsed_json = json.loads(content)
            return parsed_json.get("queries", [user_query])
        except Exception as e:
            print(f"Error during query decomposition: {e}")
            return [user_query] # Fallback to an empty list


    def decide(self, prompt: str):
        """Generates a structured Pydantic object as a decision."""
        try:
            # **NEW**: Use the patched method with response_model
            decision = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["SEARCH", "ANSWER_DIRECTLY"]
                            },
                        },
                        "required": ["action"]
                    }
                }
            )
            print(decision['choices'][0]['message']['content'])
            return json.loads(decision['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error during structured generation: {e}")
            return {"action": "SEARCH"} # Fallback


    def generate(self, system_prompt: str, prompt: str, max_new_tokens: int = 450, temp: float = 0.2) -> str:
        """Generates a response from the GGUF model."""
        try:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": "You are a helpful assistant /no_think"},
                    {"role": "user", "content": prompt}
                ]

            # print(messages)
            # response = 'dummy'
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temp,
            )
            print(json.dumps(response, indent=2, ensure_ascii=False))
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error during LLM generation: {e}"

