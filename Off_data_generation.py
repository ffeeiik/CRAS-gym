import numpy as np
import random
import pandas as pd
import re
from datetime import datetime, timedelta
import random
import json

def generate_offline_rl_data(data_dict, num_episodes):
    data = {
        "user": [],
        "budget": [],
        "reward": [],
        "flow_position": [],
        "ad_position": [],
        "time_slot": [],
        "rank_ad_num": [],
        "action": [],
        "cost": []
    }

    sofa_trace_ids = list(data_dict.keys())
    print("sofa_trace_ids length is:", len(sofa_trace_ids))
    #每个episode开始循环，每个episode循环100次
    for _ in range(num_episodes):
        start_trace_id = random.choice(sofa_trace_ids)
        start_time = data_dict[start_trace_id][0]['time']
        
        # print(start_trace_id)
        #生成每个episode的流量集合
        episode_trace_ids = []
        episode_trace_ids.append(start_trace_id)

        tmp = sofa_trace_ids.index(start_trace_id)+1
        print("start_trace_index", tmp)

        while second_difference(data_dict[start_trace_id][0]['time'], data_dict[sofa_trace_ids[tmp]][0]['time']) <= 60:
            episode_trace_ids.append(sofa_trace_ids[tmp])
            print(sofa_trace_ids[tmp])
            tmp += 1
        
        print("episode_trace_ids:", episode_trace_ids)

        #当前流量五分钟前流量数
        users = []
        for trace_id in episode_trace_ids:
            users.append(num_five_minutes_ago(data_dict, sofa_trace_ids, trace_id))

        #reward，cost读取
        for _ in range(iter_num_episode):
            flow_position = 0
            total_cost = 0
            total_reward = 0

            for trace_id in episode_trace_ids:
                actions = generate_action_queue(data_dict, trace_id)
                action = random.choice(actions)

                for item in data_dict[trace_id]:
                    if item['sim_instance_id'] == action:
                        reward = item['new_sum_ecpm']
                        cost = item['cpu_flops']
                        break

                total_cost += cost
                total_reward += reward

                data["user"].append(users[flow_position])
                data["budget"].append(max_total_cost - total_cost)
                data["reward"].append(reward)
                data["flow_position"].append(flow_position)
                data["ad_position"].append(item['ad_pos_id'])
                data["time_slot"].append(time2minute(item['time']))
                data["rank_ad_num"].append(item['rank_ad_num'])
                data["action"].append(action)
                data["cost"].append(cost)

            
    return data


    #action生成，取每条流量前5名reward的action进行组合
def generate_action_queue(data_dict, current_trace_id):
    actions = []
    sim_instances = []

# 获取当前流量的所有sim_instance_id
    for item in data_dict[current_trace_id]:
        sim_instances.append(item['sim_instance_id'])

            # 根据new_sum_ecpm排序，取前5个值对应的action
    sorted_items = sorted(data_dict[current_trace_id], key=lambda x: x['new_sum_ecpm'], reverse=True)
    for item in sorted_items[:5]:
        actions.append(item['sim_instance_id'])

    return actions
        

def second_difference(current_time, next_time):
    current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    next_time = datetime.strptime(next_time, '%Y-%m-%d %H:%M:%S')
    time_difference = next_time - current_time
    second_difference = time_difference.total_seconds()

    return second_difference
    
def time2minute(current_time):
    datetime_obj = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    current_minute = datetime_obj.hour * 60 + datetime_obj.minute
        
    return current_minute

            #当前流量前5分钟的流量数
def num_five_minutes_ago(data_dict, sofa_trace_ids, current_trace_id):
    current_time = data_dict[current_trace_id][0]['time']
    current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    five_minutes_ago = current_time - timedelta(minutes=5)
    tmp = sofa_trace_ids.index(current_trace_id) - 1
    trace_id = datetime.strptime(data_dict[sofa_trace_ids[tmp]][0]['time'], '%Y-%m-%d %H:%M:%S')
    users = 0
    while trace_id >= five_minutes_ago:
        tmp -= 1
        trace_id = datetime.strptime(data_dict[sofa_trace_ids[tmp]][0]['time'], '%Y-%m-%d %H:%M:%S')
        users += 1
    return users



# 调用原代码中的函数和方法
if __name__ == "__main__":


    num_episodes = 100
    iter_num_episode = 100
    max_total_cost = 40000

    #读取数据
    data = pd.read_csv('data/test_train_table_0514.csv')
    data = data[10000:20000]
    # 打印数据
    # print(data)
    data_dict = {}
    for sofa_trace_id, group in data.groupby('sofa_trace_id'):
        group_dict = group.drop('sofa_trace_id', axis=1).to_dict(orient='records')
        data_dict[sofa_trace_id] = group_dict
    # print("dict length is:",len(list(data_dict.keys())))
    data_dict = dict(sorted(data_dict.items(), key=lambda x: min(x[1], key=lambda y: datetime.strptime(y['time'], '%Y-%m-%d %H:%M:%S'))['time']))
    # print("dict length is:",len(list(data_dict.keys())))
    for key,value in data_dict.items():
        for item in value:
            item['sim_instance_id'] = re.sub(r'step_(\d+)', lambda x: str(int(x.group(1))), item['sim_instance_id'])

        data_dict[key] = sorted(value, key=lambda x: int(x['sim_instance_id']))

    # 打印整理后的字典
    # print(data_dict)

    data = generate_offline_rl_data(data_dict, num_episodes)


# 将字典保存为JSON文件
    with open('output.json', 'w') as f:
        json.dump(data, f)



