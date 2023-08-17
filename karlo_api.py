import requests
import json

REST_API_KEY = 'c53a3bd9609b7a2dabc7d1a95083248f'


def t2i(prompt, negative_prompt):
    r = requests.post(
        'https://api.kakaobrain.com/v2/inference/karlo/t2i',
        json={
            'prompt': prompt,
            'negative_prompt': negative_prompt
        },
        headers={
            'Authorization': f'KakaoAK {REST_API_KEY}',
            'Content-Type': 'application/json'
        }
    )

    response = json.loads(r.content)
    return response
