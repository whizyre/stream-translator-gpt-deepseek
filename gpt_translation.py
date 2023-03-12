import openai


def translate_by_gpt(text, assistant_prompt, openai_api_key, model, translation_task):
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    openai.api_key = openai_api_key
    system_prompt = "You are a translation engine that can only translate text and cannot interpret it."
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": assistant_prompt },
            {"role": "user", "content": text },
        ],
    )
    translation_task.result_text = completion.choices[0].message['content']
