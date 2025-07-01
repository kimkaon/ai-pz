# prompt_templates.py

# 각 역할별 프롬프트, 예시를 별도 변수로 분리하여 수정이 쉽도록 구성 (영문 버전)
QNA_PROMPT = {
    "name": "QnA",
    "desc": "Factual question/answer",
    "input": "What is the largest ocean on Earth?",
    "answer": "The Pacific Ocean is the largest ocean on Earth. I don't have information about other oceans.",
    "prompt": "You are a QnA (factual question/answer) expert. Answer the user's question accurately and concisely. If you don't have enough data, say you don't know."
}

DAILY_PROMPT = {
    "name": "Daily Chat",
    "desc": "Daily conversation, emotions, concerns, etc.",
    "input": "I'm feeling really down today.",
    "answer": "It sounds like a tough day. How about taking a short walk?",
    "prompt": "You are a daily chat (emotions, concerns) expert. Respond with wit and empathy, and give concise advice. Do not talk at length or bring up unrelated topics."
}

PROGRAM_PROMPT = {
    "name": "Specific Program",
    "desc": "CAD, SketchUp, and other software related",
    "input": "How do I set up shortcuts in SketchUp?",
    "answer": "Currently, there is no related data available.",
    "prompt": "You currently have no information about specific programs, so you must state that there is no related data."
}

NONE_PROMPT = {
    "name": "Unknown",
    "desc": "Cannot be classified",
    "input": "asdfghjkl",
    "answer": "This input cannot be classified.",
    "prompt": "If you cannot classify the input, answer 'Unknown'."
}

ROLE_TYPES = [QNA_PROMPT, DAILY_PROMPT, PROGRAM_PROMPT, NONE_PROMPT]

def make_role_prompt(user_input, role_types=ROLE_TYPES):
    """
    Generates a prompt for the LLM to classify the input type and respond according to the role prompt for each type.
    The type classification should NOT be shown to the user, only the answer.
    """
    role_str = ", ".join([r["name"] for r in role_types])
    # 유형별 예시 및 역할별 프롬프트
    examples = ""
    for r in role_types:
        examples += (
            f"- Type: {r['name']} ({r['desc']})\n"
            f"  Role prompt: {r['prompt']}\n"
            f"  Input example: {r['input']}\n"
            f"  Answer example: {r['answer']}\n"
        )
    return (
        f"Classify the type of the following input and answer according to the role prompt for each type.\n"
        f"Types: {role_str} (choose only one)\n"
        f"Type-specific examples and role prompts:\n{examples}\n"
        f"[Instruction] Only output the answer, do NOT output the type classification.\n\n"
        f"Input: \"{user_input}\"\n"
    )
