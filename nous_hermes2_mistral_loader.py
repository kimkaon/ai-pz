from llama_cpp import Llama

def load_nous_hermes2_mistral(model_path="models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf", n_ctx=8192, n_gpu_layers=32):
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return llm

def chat_nous_hermes2_mistral(llm, prompt):
    output = llm(prompt, max_tokens=512, stop=["</s>"])
    return output["choices"][0]["text"].strip()
