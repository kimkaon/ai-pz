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

def chat_nous_hermes2_mistral_stream(llm, prompt):
    """
    실시간 스트리밍으로 토큰을 하나씩 생성하여 반환
    """
    try:
        for token in llm(prompt, max_tokens=512, stop=["</s>"], stream=True):
            if token and "choices" in token and len(token["choices"]) > 0:
                text = token["choices"][0]["text"]
                if text:
                    yield text
    except Exception as e:
        print(f"스트리밍 오류: {e}")
        # 스트리밍 실패 시 일반 방식으로 fallback
        output = llm(prompt, max_tokens=512, stop=["</s>"])
        yield output["choices"][0]["text"].strip()
