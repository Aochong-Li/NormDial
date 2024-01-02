import pandas as pd
import numpy as np
import json
import openai
import pickle
import time
import re
import os
from transformers import AutoTokenizer, AutoModel

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_api (api, base_msg = None, input_msg = None, temp = 1.0, max_tokens = 1024):
    openai.api_key = api
    model = "gpt-3.5-turbo"
    
    completion = openai.ChatCompletion.create(
        model = model,
        max_tokens = max_tokens,
        temperature = temp,
        messages=[
            {"role": "system", "content": base_msg},
            {"role": "user", "content": input_msg}
        ]
    )
    
    return completion

def generate_dialogue(api, categories, input_PATH, output_PATH):

    with open(input_PATH, 'rb') as f:
        norm_situation_list = pickle.load(f)
    
    for key in categories:
        print('Generating Dialogues for {} ...'.format(key))

        norm_situation_df = norm_situation_list[key]
        dialogue_list = []

        for i in range(len(norm_situation_df)):
            norm = norm_situation_df['cn_Norm'][i]
            situation = norm_situation_df['cn_Situation'][i]

            prompt = '''每次请根据一个围绕着中国社会规范的生活情景，有创意地生成一段人物间真实自然的对话脚本。

要求:
1. 对话中提及情景中所有细节和内容
2. 只需要生成对话脚本不需要额外解释
3. 且请以“对话” 为开头生成对话，以“[结束]”标注对话结尾。

规范：如果你妨碍到了另一个人，你应该道歉并且询问对方以表示关心
情境：中国新年期间，一个中国小伙子大伟在王府井街上不小心撞到了纽约来找朋友的女人苏珊，大伟多次询问了苏珊是否受伤表示关心并且多次道歉。苏珊也同样地询问大伟是否被妨碍到，并且大伟因看到苏珊作为美国人说中文表示很新奇。

对话
大伟和苏珊: 哎呀
大伟: 哎呦，对不起，没撞到您吧
苏珊: 没事没事，真对不起
大伟: 没想到您还说中国话呢，您好
苏珊: 你好
大伟:  我刚才没碰到你吧?
苏珊: 我很好，就是不会走路，你还好吗
大伟: 我没事，新年快乐，注意安全
[结束]'''
            input = norm + '\n'+ situation
            prompt += '\n\n' + input

            base_msg = '''You are a culture-aware system that can generator natural dialogues in Chinese'''

            response = gpt_api(api = api, base_msg = base_msg , input_msg = prompt)
            response = response['choices'][0]['message']['content']

            dialogue_list.append(response)
        
        norm_situation_list[key]['Dialogue_ChatGLM'] = dialogue_list

        if os.path.isdir(output_PATH+'/{}'.format(key)) != True:
            os.mkdir(output_PATH+'/{}'.format(key))

        norm_situation_list[key].to_csv(output_PATH+'/{}'.format(key)+'/dialogue.csv', index=False)
        print('Saving Dialogues for {} ...'.format(key))

if __name__=='__main__':
    
    process_id = 3

    APIs = [
        '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',
        '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',
        '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',
        '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',
        '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'
    ]
    
    categories = [
        # ['criticism', 'compliments'],
        ['respond_compliments','thanks'],
        ['criticism', 'persuasion'],
        ['leave'],
        ['condolence', 'greeting'],
        # ['thanks', 'leave']
    ]

    input_PATH = '/home/al4143/norm/output_ChatGPT/ChatGPT_dialogue/extra_violation_label/norm_situations.pickle'
    output_PATH = '/home/al4143/norm/output_ChatGPT/ChatGPT_dialogue/extra_violation_label/dialogue'
    generate_dialogue(api = APIs[process_id], categories = categories[process_id], input_PATH = input_PATH, output_PATH = output_PATH)

    # with open('/home/al4143/norm/output_ChatGPT/dialogue/adherence/norm_situation_dialogue.pickle', 'wb') as f:
    #     pickle.dump(norm_situation_dialogue_list, f, protocol=pickle.HIGHEST_PROTOCOL)

