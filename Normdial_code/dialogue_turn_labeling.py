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
def gpt_api (api, msg = None, temp = 0.1, max_tokens = 1024):
    openai.api_key = api
    model = "gpt-3.5-turbo"
    
    completion = openai.ChatCompletion.create(
        model = model,
        max_tokens = max_tokens,
        temperature = temp,
        messages = msg
    )
    
    return completion

def preprocess_input (dialogue):

    if dialogue[:2] == '对话':
        dialogue = dialogue[2:]

    if '[结束]' in dialogue:
        dialogue = dialogue.replace('[结束]', '')
    
    dialogue = 'Dialogue:' + dialogue
    
    return dialogue

def model_input (norm, dialogue):
    base_msg = ''

    user_input = '''Given a dialogue and a norm which the dialogue is based on, the task has 4 steps:

1. Summarize the Norm in 5 words as Norm Action
2. Indicate which character in the dialogue performs the Norm Action
3. Repeat every turn and only indicate 'Adhered' when the sentence closely aligns with the Norm. Otherwise, indicate 'Not Relevant'. 
4. After each label, provide a short explanation for why the norm is strongly entailed  or not relevant at sentence level.

Format:
Repeat each turn in a bracket
Append Adhered or Not Relevant label for each turn
Use “|” to separate role, label and explanation if needed

Norm: In a professional setting with higher status speaking to lower status, it is permitted to use direct language, a strong tone of voice, and display emotions when criticizing one's behavior, ideas, and work.

Dialogue:
张教练: 小陈，进来坐。你今天比赛时的那个失误，不止是你自己比赛历史有了污点，也让我们队失去了比赛胜利的机会。
小陈: 我知道我做错了。
张教练: 而且我强调的不仅仅是你犯的错，而是你没有注意到你思想问题。
张教练: 小陈，你需要更多的多传球给你的队友，不能老是单打独斗。
小陈: 教练我会改正的
张教练: 你今天的投篮还是很不错的，继续努力。
小陈: 谢谢教练。我一定会好好听取你的建议。
张教练: 很好，去休息吧。'''
    
    response = '''Norm Action: offer criticism 

Actor of the Norm:
张教练:  coach, higher status, criticizer

Dialogue:
(张教练: 小陈，进来坐。你今天比赛时的那个失误，不止是你自己比赛历史有了污点，也让我们队失去了比赛胜利的机会。):  Adhered | 张教练 criticizes his player’s performance by using direct wordings including "失误", "污点", and "让我们队伍失去"
(小陈: 我知道我做错了。): Not Relevant | 小陈 is not acting the criticism norm
(张教练: 而且我强调的不仅仅是你犯的错，而是你没有注意到你思想问题。): Adhered | 张教练 criticizes 小陈’s ideas of how to play basketball by questioning him
(张教练: 小陈，你需要更多的多传球给你的队友，不能老是单打独斗。): Adhered | 张教练 offers a mild criticism by saying “不能老师单打独斗”
(小陈: 教练我会改正的):   Not Relevant | 小陈 is not an actor of criticism norm
(张教练: 你今天的投篮还是很不错的，继续努力。): Not Relevant | 张教练 does not criticize here
(小陈: 谢谢教练。我一定会好好听取你的建议。): Not Relevant ｜小陈 is not a criticizer 
(张教练: 很好，去休息吧。): Not Relevant | not criticism statement'''

    msg = [
        {"role": "system", "content": base_msg},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
        {"role": "user", "content": norm + '\n\n' + dialogue}
    ]

    return msg

def labeling (API, input_PATH, output_PATH, categories):
    
    for category in categories:
        PATH = input_PATH + '/{}'.format(category) + '/dialogue.csv'
        
        dialogue_df = pd.read_csv(PATH)
        labeling = []
        
        print('Label Dialogues for {} ...'.format(category))

        for i in range(len(dialogue_df)):
            dialogue = dialogue_df['Dialogue'][i]
            norm = dialogue_df['Norm'][i]

            dialogue = preprocess_input(dialogue)
            input_msg = model_input(norm, dialogue)
            response = gpt_api(api = API, msg = input_msg)
            response = response['choices'][0]['message']['content']

            labeling.append(response)
        
        dialogue_df['Labels_ChatGPT'] = labeling

        ### Save the labels

        if os.path.isdir(output_PATH+'/{}'.format(category)) != True:
            os.mkdir(output_PATH+'/{}'.format(category))

        dialogue_df.to_csv(output_PATH+'/{}'.format(category)+'/labeled_dialogue.csv', index=False)
        print('Saving Labels for {} ...'.format(category))
            

if __name__=='__main__':
    process_id = 4

    APIs = [
        'YOUR OPENAI API KEYS'
    ]
    
    categories = [
        ['condolence', 'respond_compliments'],
        ['greeting', 'thanks'],
        ['criticism', 'persuasion'],
        ['leave'],
        # ['thanks', 'leave']
    ]

    input_PATH = 'INPUT PATH for dialogue directory'
    output_PATH = 'OUTPUT PATH for labeled dialogues'

    labeling(API = APIs[process_id], input_PATH = input_PATH, output_PATH = output_PATH, categories = categories[process_id])