from openai import OpenAI
client = OpenAI(
    base_url='https://api.key77qiqi.cn/v1',
    api_key='sk-qoc1zRByuvCzlrZFgBaA6wqRwkUbyHQUUXnIKXgNXnbIGOXj'
    )#Setting by yourself
prompt_text=[{"type": "text", "text": f'hello'}]
# input={"messages":prompt_text}

response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ]
            )
print(response.choices[0].message.content.strip())