# prompt_templates.py

# 통일된 기본 프롬프트 (모든 모델이 일관되게 사용)
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Respond naturally and directly to the user's input without any prefixes, labels, or classifications. 

Important guidelines:
- DO NOT start your response with "AI:", "Assistant:", or any role labels
- DO NOT include type classifications like "[QnA]", "[Daily Chat]", etc.
- DO NOT repeat the user's question in your answer
- Respond concisely and naturally as if you're having a normal conversation
- If you don't know something, simply say you don't know

Examples of good responses:
User: "What is the largest ocean?"
Good: "The Pacific Ocean is the largest ocean on Earth."
Bad: "AI: [QnA] The Pacific Ocean is the largest ocean on Earth."

User: "I'm feeling down today."
Good: "I'm sorry to hear that. Maybe taking a short walk or talking to someone you trust could help."
Bad: "Assistant: [Daily Chat] It sounds like you're having a tough day..."
"""

# 레거시 프롬프트 (기존 호환성 유지)
QNA_PROMPT = {
    "name": "QnA",
    "desc": "Factual question/answer",
    "input": "What is the largest ocean on Earth?",
    "answer": "The Pacific Ocean is the largest ocean on Earth.",
    "prompt": "Answer the user's question accurately and concisely."
}

DAILY_PROMPT = {
    "name": "Daily Chat",
    "desc": "Daily conversation, emotions, concerns, etc.",
    "input": "I'm feeling really down today.",
    "answer": "I'm sorry to hear that. Maybe taking a short walk could help.",
    "prompt": "Respond with empathy and give helpful advice."
}

PROGRAM_PROMPT = {
    "name": "Specific Program",
    "desc": "CAD, SketchUp, and other software related",
    "input": "How do I set up shortcuts in SketchUp?",
    "answer": "I don't have specific information about SketchUp shortcuts available right now.",
    "prompt": "If you don't have specific information, say so politely."
}

NONE_PROMPT = {
    "name": "Unknown",
    "desc": "Cannot be classified",
    "input": "asdfghjkl",
    "answer": "I'm not sure what you mean. Could you clarify your question?",
    "prompt": "Ask for clarification politely."
}

ROLE_TYPES = [QNA_PROMPT, DAILY_PROMPT, PROGRAM_PROMPT, NONE_PROMPT]

def make_default_prompt(user_input):
    """
    새로운 통일된 기본 프롬프트 생성 함수
    모든 모델이 일관되게 깔끔한 응답을 하도록 함
    """
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"

def make_role_prompt(user_input, role_types=ROLE_TYPES):
    """
    레거시 역할 기반 프롬프트 (기존 호환성 유지)
    하지만 이제 기본 프롬프트를 사용하는 것을 권장
    """
    # 기본 프롬프트 사용으로 변경
    return make_default_prompt(user_input)
