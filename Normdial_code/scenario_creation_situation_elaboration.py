import pandas as pd
import numpy as np
import json
import openai
import pickle
import time
import re

def gpt_api (base_msg = '', input_msg = '', custom_msg = None):
    openai.api_key = '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'
    model = "gpt-3.5-turbo"
    
    
    if custom_msg != None:
        completion = openai.ChatCompletion.create(
            model = model,
            max_tokens = 1024,
            temperature = 1.0,
            messages= custom_msg
        )
    else:
        completion = openai.ChatCompletion.create(
        model = model,
        max_tokens = 1024,
        temperature = 1.0,
        messages=[
            {"role": "system", "content": base_msg},
            {"role": "user", "content": input_msg}
        ]
    )
    
    return completion

def scenario_generation(prompts):

    one_shot = """Social norms are informal rules that govern behaviors in groups and societies. You are given a social norm, and you are tasked to imagine and briefly describe 10 scenarios that a conversation that entails the Norm can take place in a real-life setting of Chinese society. 
    
    Format: Start with your response with “Scenario:”.

    Norm: It is socially preferred to apologize immediately if you disturb another person and give the affected person a chance to identify and specify if they are hurt

    Scenario:
    1. in a university; college students
    2. on the street; strangers
    3. in a company’s office; colleagues
    4. in a hospital; patient and doctors
    5. in a restaurant; waiter and customers 
    6. in a cafe; two customers
    7. in a shopping mall; sales associates and customers 
    8. in a park; a morning jogger and a lady
    9. in a suburb neighborhood; two neighbors who know each other
    10. in a family gathering; two cousins
    """

    prompts = ["Norm: " + prompt for prompt in prompts]
    scenario_df  = pd.DataFrame()
    scenario_df['prompts'] = prompts
    prompts = [one_shot + '\n' + prompt for prompt in prompts]
    responses  = []
    
    ##Call GPT
    base_msg = "You are a culture-aware system that can generate real-world scenarios in Chinese society."

    for i in range(len(prompts)):
        try:
            response = gpt_api(base_msg = base_msg, input_msg = prompts[i])
            response = response['choices'][0]['message']['content']
            responses.append(response)
        except Exception as e:
            print(prompts[i])
            print(e)
        
        time.sleep(1)

    scenario_df['scenario'] = responses
    
    return scenario_df
    
def situation_expansion(norm, scenes):
    one_shot = '''Social norms are informal rules that govern behaviors in groups and societies. You are given a situation and social norm, and you are tasked to create and include details to the real-life situation which takes place in Chinese society.

    Format: start with “New Situation:”. 

    Norm: It is socially preferred to apologize immediately if you disturb another person and give the affected person a chance to identify and specify if they are hurt
    Situation: On the street; a Chinese young man and a woman 

    New Situation:  A Chinese young man, 大伟, on his way back home, bumped into a stranger named Susan on the street. Susan is from New York, America, and it is her first time coming to China looking for her friend, so she doesn’t speak fluent Chinese and is lost on the street.
    '''

    if "Scenario:" not in scenes:
        raise Exception('No Scenario in the input')
    
    ### Preprocess situation
    scenes = scenes.split('Scenario:')[1]
    scenes = scenes.split('\n')
    try:
        scenes  = [scene.replace(re.findall(r'\b([0-9]|10)\b', scene)[0]+'.', '') for scene in scenes if len(re.findall(r'\b([0-9]|10)\b', scene))!=0]
    except Exception as e:
        print('Fail to parse scenarios')
        print(scenes)

    ### Prepare prompts
    prompts = [norm+'\nSituation:'+scene for scene in scenes]
    prompts = [one_shot + '\n' + prompt for prompt in prompts]    

    ### Output
    situation_out = dict()
    situation_out["Norm"]= norm
    situation_out['Situation']  = []
    
    
    base_msg = "You are culture-aware system that can elaborate on a given situation."
    ### Getting Responses
    for i in range(len(prompts)):
        try:
            response = gpt_api(base_msg = base_msg, input_msg = prompts[i])
            response = response['choices'][0]['message']['content']
            start = response.index('New Situation')
            response = response[start:]
            response = response.replace('New Situation', 'Situation')

            situation_out['Situation'].append(response)
        except Exception as e:
            print(prompts[i])
            print(e)
        
        time.sleep(1)
            
    return situation_out

def situation_expansion_wrapper(scenario_df):
    expanded_situation = []
    
    for i in range(len(scenario_df)):
        completion = situation_expansion(scenario_df['prompts'][i],scenario_df['scenario'][i])
        expanded_situation.append(completion)
    
    return expanded_situation

def Label(norm, dialogue):

    user1 = '''You are given a dialogue and a norm that appears in the dialogue. 

        Repeat the exact dialogue turn by turn. For each turn indicate Adhere if the sentence highly entails the norm and Not Relevant if not.

        Format: 
        Start your response with “Dialogue:”
        Append Adhere or Not Relevant Label at the end of each turn
        Use “|” to separate each sentence and Adhere label

        Norm: In a professional setting with higher status speaking to lower status, it is permitted to use antonomasia to do the greeting. A higher status individual will directly call the lower status individual without using honorifics

        Roles:
        Dr. Chen: higher-status individual
        Wei: lower-status individual

        Dialogue:
        陈教授: 魏欢迎加入我们 | Adhere
        魏: 您好，感谢您 | Not Relevant
        陈教授: 没有福利但是工作很有价值 | Not Relevant
        魏: 好的, 谢谢您的说明 | Not Relevant
        陈教授: 你有什么问题吗？ | Adhere
        魏: 这个职位的工作时间是多长？ | Not Relevant
        陈教授: 大概是每周十个小时, 主要是协助课程的辅助工作和研究 | Not Relevant
        魏: 好的，我明白了, 非常感谢您 | Not Relevant

        Norm: In a formal or professional setting or with people of a higher status, it is obligatory to offer thanks directly when a lower status person is talking to a higher status person. Thanks from a higher status person to a lower status person is optional.

        Roles:
        Ming: lower status person
        Professor: higher status person

        Dialogue:
        明: 老师，我有一个问题想请教您
        教授: 请讲
        明: 请问这个数学题为什么要这么做？
        教授: 好，这是一个很好的问题。这个题目的解法其实是。。。
        明: 谢谢老师，我明白了
        教授: 不用谢，你有什么问题以后还可以问我'''

    assistant1 = '''Dialogue:
        明: 老师，我有一个问题想请教您 | Adhere
        教授: 请讲 | Not Relevant
        明: 请问这个数学题为什么要这么做？ | Not Relevant
        教授: 好，这是一个很好的问题。这个题目的解法其实是。。。| Not Relevant
        明: 谢谢老师，我明白了 | Adhere
        教授: 不用谢，你有什么问题以后还可以问我 | Not Relevant'''

    user2 = '''The first sentence is not about giving thanks, unrelated to the Norm. So the label is wrong. Repeat the Dialogue with correct Labels'''
    assistant2 = '''Dialogue:
        明: 老师，我有一个问题想请教您 | Not Relevant
        教授: 请讲 | Not Relevant
        明: 请问这个数学题为什么要这么做？ | Not Relevant
        教授: 好，这是一个很好的问题。这个题目的解法其实是。。。| Not Relevant
        明: 谢谢老师，我明白了 | Adhere
        教授: 不用谢，你有什么问题以后还可以问我 | Not Relevant'''

    user3 = '''Norm: In a situation where a minor offense such as bumping into someone or arriving late to a meeting has been committed, it is preferable to give a mild apology to acknowledge the slight wrong.

        Roles:
        Wei: apologizing person and offender
        Curator: affected party

        Dialogue:
        魏: 哎呦，对不起，我没看到，我真是太粗心了
        馆长: 哎呀，没关系，这个陈列品也快到期了，不影响整个展览，你不用太自责了
        魏: 不不不，我真是万分抱歉，我能为我犯的错误承担责任，我愿意支付赔偿款
        馆长: 不用了，我认为你的态度真的很好，很诚恳，会让人感到舒服的。对不起，这样的艺术品比较脆弱，希望你以后会注意一些。'''

    assistant3 = '''Dialogue:
        魏: 哎呦，对不起，我没看到，我真是太粗心了 | Adhere
        馆长: 哎呀，没关系，这个陈列品也快到期了，不影响整个展览，你不用太自责了 | Adhere
        魏: 不不不，我真是万分抱歉，我能为我犯的错误承担责任，我愿意支付赔偿款 | Adhere
        馆长: 不用了，我认为你的态度真的很好，很诚恳，会让人感到舒服的。对不起，这样的艺术品比较脆弱，希望你以后会注意一些。 | Not Relevant'''

    user4 = '''The second sentence is not about giving an apology and thus unrelated to the norm. The label is wrong. Repeat the Dialogue with correct Labels'''
    assistant4 = '''Dialogue:
        魏: 哎呦，对不起，我没看到，我真是太粗心了 | Adhere
        馆长: 哎呀，没关系，这个陈列品也快到期了，不影响整个展览，你不用太自责了 | Not Relevant
        魏: 不不不，我真是万分抱歉，我能为我犯的错误承担责任，我愿意支付赔偿款 | Adhere
        馆长: 不用了，我认为你的态度真的很好，很诚恳，会让人感到舒服的。对不起，这样的艺术品比较脆弱，希望你以后会注意一些。 | Not Relevant'''
    
    if 'Explanation' in dialogue:
        idx = dialogue.index('Explanation')
        dialogue = dialogue[:idx]

    prompt =  norm + '\n' + dialogue

    custom_msg  = [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
            {"role": "assistant", "content": assistant2},
            {"role": "user", "content": user3},
            {"role": "assistant", "content": assistant3},
            {"role": "user", "content": user4},
            {"role": "assistant", "content": assistant4},
            {"role": "user", "content": prompt}
        ]
    
    response = gpt_api(custom_msg = custom_msg)
    labeled_dialogue = response['choices'][0]['message']['content']
    
    return labeled_dialogue

def ProduceDialogue (norm_situation_dic):

    shots = """Your task is to generate a natural dialogue in Chinese around the given social norm. Social norms are informal rules that govern behaviors in groups and societies.

    The task has 3 steps:
    1. Indicate the roles each character takes in the Norm. Start with “Roles:”
    2. Create a dialogue of at least 5 turns between characters that highly entails the norm and includes the details about the situation. Start with “Dialogue:”
    3. Based on the dialogue, briefly explain how the Norm is performed. Start with “Explanation:”

    Norm: It is socially preferred to apologize immediately if you disturb another person and give the affected person a chance to identify and specify if they are hurt
    Situation: A Chinese young man Dawei, who is taunting his nephew, accidentally bumps into a woman Susan on the street

    Roles:
    Dawei: apologizing person and affected party
    Susan: apologizing person and affected party

    Dialogue: 
    大伟和苏珊: 哎呀
    大伟: 哎呦，对不起，没撞到您吧
    苏珊: 没事没事，我叫苏珊，你叫什么？
    大伟: 还说中国话呢, 您好，我叫大伟
    苏珊: 你好
    大伟: 你好吗? 我刚才没碰到你吧?
    苏珊: 我很好, 就是不会走路 
    大伟: 您什么意思？我没太听懂
    苏珊: 抱歉，我说错了。我的意思是没找到路。
    大伟: 啊, 我刚才没踩到您的脚吧 
    苏珊: 没事没事，您没受伤吧
    大伟: 我没关系，真是抱歉哈


    Explanation: Dawei and Susan apologizes to each other using honorifics as described in the norm. Dawei first asks Susan if she is hurt and shows his concerns for Susani, who does the same thing later.

    Norm: In a setting where a stranger wants to approach another stranger’s belongings, it is obligatory to request permission by addressing them with honorific you 您 and using please 请
    Situation: A passenger Wang who asks permission to get into the car of an Uber driver Li whom she does not know and tells the driver her destination.

    Roles: 
    Wang: person who approaches another’s belonging
    Li: person who owns the belonging

    Dialogue:
    李司机: 您好
    王小姐: 请问是您的车吗，网上约的Uber司机
    李司机: 对对对， 您上来吧
    王小姐: 我到人民医院，在北四环建安路上
    李司机: 嗯嗯好好，请您系好安全带
    王小姐: 麻烦师傅您了啊
    李司机: 没事没事，那咱们出发了

    Explanation: Wang, as a passenger, first politely asks if she is permitted to come into Li’s car by indirectly asking “请问是您的车吗.” Then, Wang thanks Li for carrying her by addressing him as “师傅” and “您.”
    """

    ### Preprocess Norm and Situation
    norm = norm_situation_dic['Norm']
    situations = norm_situation_dic['Situation']

    ### Prepare Prompt
    prompts = [norm+'\n'+situation for situation in situations]
    prompts = [shots + '\n\n' + prompt for prompt in prompts]

    ### Ouput
    conversation_df = pd.DataFrame(columns = ['Norm', 'Situation', 'Dialogue', 'Labeled Dialogue'])
    
    ### Call GPT
    base_msg = "You are a culture-aware system that can generate smooth Chinese dialogues in a real-life setting."

    ### Getting Response
    for i in range(len(prompts)):
        try:
            response = gpt_api(base_msg = base_msg, input_msg = prompts[i])
            dialogue = response['choices'][0]['message']['content']
            labeled_dialogue = Label(norm, dialogue)
            conversation_df.loc[len(conversation_df)] = [norm, situations[i], dialogue, labeled_dialogue]
        except Exception as e:
            print(prompts[i])
            print(e)
        
        time.sleep(1)

    return conversation_df


def dialogue(norms, OUTPUT_PATH):
    
    try:
        print('Generating Scenarios ...')
        scenario_df = scenario_generation(norms)
        scenario_df.to_csv(OUTPUT_PATH+'/scenarios.csv', index=False)
    except Exception as e:
        print(e)
        return None

    
    try:
        print('Expanding Situations...')
        expanded_situations = situation_expansion_wrapper(scenario_df)
        with open(OUTPUT_PATH + '/expanded_sitautions.pickle', 'wb') as f:
            pickle.dump(expanded_situations, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)
        return None

    print('Generating Dialogues...')
    dialogue_list = []
    
    for i in range(len(expanded_situations)):
        try:
            dialogue_df = ProduceDialogue(expanded_situations[i])
            dialogue_list.append(dialogue_df)
        except Exception as e:
            print(e)
    
    dialogues = pd.concat(dialogue_list, ignore_index=True)
    
    return dialogues

if __name__=='__main__':
    
    norms = [
    'When a person of lower status respond to the compliments of one of high status, one can show embarrassment 我都不好意思了.',
    'When a person of lower status respond to the compliments of one of high status, one can appreciate the leadership 谢谢领导的栽培,是您的功劳.',
    'Offering condolences is an important social norm when someone has experienced the loss of a loved one. It demonstrates respect and support for the grieving family.',
    'Avoiding small talk: When doing condolence, it is important to avoid small talk and stay focused on the purpose of the conversation. You may say something like "今天我来是为了向你们表达我的慰问".',
    'When a person commonly associated with a group deeply offends another person who has a relationship with that group, but the offender is not necessarily sorry for it, it is obligatory that a person associated with the offender’s group, especially a person who bears responsibility, extend an apology to the offended party on behalf of the offender. It is common for the person to accept personal responsibility even though they were not the offender.'
    ]

    OUTPUT_PATH = '/home/al4143/norm/output/experiment'

    dialogues = dialogue(norms, OUTPUT_PATH)

    dialogues.to_csv(OUTPUT_PATH+'/dialogue.csv', index=False)

    