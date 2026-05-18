import json
import os
from datetime import datetime, timedelta
import argparse

def get_task_path(task_id):
    """根据传入的task_id参数返回对应的文件路径"""
    
    lamp_tasks = {
        "1": "citation",                   
        "2M": "movie_tagging",             
        "2N": "news_categorize",           
        "4": "news_headline",              
        "5": "product_rating",             
        "6": "scholarly_title",            
        "7": "tweet_paraphrase"            
    }

    task_name = lamp_tasks[task_id]
    
    # 生成文件路径
    input_file = f'./data/{task_name}/user_top_100_history.json'
    
    return input_file, task_name
def find_profile_split_point(profile):
    """
    自动找到profile中的分割点
    分割点定义为第一个用户profile的最后一项ID的位置
    """
    if not profile:
        return 0
    
    # 使用第一个用户profile的最后一项的ID作为查找条件
    last_id_of_first_user = profile[-1].get('id')
    for i, item in enumerate(profile):
        if item.get('id') == last_id_of_first_user and i != len(profile) - 1:
            return i + 1
    return len(profile) // 2

def find_query_split_point(query):
    """
    自动找到query中的分割点
    分割点定义为第一个用户query的最后一项ID的位置
    """
    if not query:
        return 0
    
    # 使用第一个用户query的最后一项的ID作为查找条件
    last_id_of_first_user = query[-1].get('id')
    for i, item in enumerate(query):
        if item.get('id') == last_id_of_first_user and i != len(query) - 1:
            return i + 1
    return len(query) // 2

def adjust_timestamps(profile_part1, profile_part2):
    """
    调整第二个部分的时间戳，使其在第一个部分之后顺延
    """
    if not profile_part1 or not profile_part2:
        return profile_part1, profile_part2
    
    # 获取第一部分最后一条记录的时间戳
    last_item = profile_part1[-1]
    last_time_str = last_item.get('date') or last_item.get('time')
    
    if not last_time_str:
        return profile_part1, profile_part2
    
    try:
        # 解析时间戳
        if 'T' in last_time_str:
            last_time = datetime.fromisoformat(last_time_str.replace('Z', '+00:00'))
        else:
            # 尝试不同的日期格式
            try:
                last_time = datetime.strptime(last_time_str, '%Y-%m-%d')
            except ValueError:
                last_time = datetime.strptime(last_time_str, '%Y/%m/%d')
        
        # 为第二部分的每条记录分配新的时间戳
        adjusted_part2 = []
        for i, item in enumerate(profile_part2):
            new_item = item.copy()
            # 新时间戳在最后时间基础上顺延天数
            new_time = last_time + timedelta(days=i+1)
            # 保持与原数据相同的日期格式
            new_item['date'] = new_time.strftime('%Y-%m-%d')
            adjusted_part2.append(new_item)
        
        return profile_part1, adjusted_part2
    except Exception as e:
        print(f"时间戳调整出错: {e}")
        return profile_part1, profile_part2

def mix_users_data(first_user, other_users):
    """
    混合第一个用户和其他用户的数据
    """
    mixed_dataset = []
    
    # 获取第一个用户最后一条记录的时间戳
    first_user_last_date = first_user['profile'][-1]['date']
    first_user_last_datetime = datetime.strptime(first_user_last_date, '%Y-%m-%d')
    
    for other_user in other_users:
        # 调整第二个用户的时间戳，使其从第一个用户最后一条记录之后开始
        adjusted_profile = []
        
        # 先添加第一个用户的profile（保持原样）
        adjusted_profile.extend(first_user['profile'])
        
        # 然后处理第二个用户的profile，调整时间戳
        second_user_profile = other_user['profile']
        for j, item in enumerate(second_user_profile):
            # 计算新的日期：在第一个用户最后日期基础上顺延
            new_date = first_user_last_datetime + timedelta(days=j+1)
            new_item = item.copy()
            new_item['date'] = new_date.strftime('%Y-%m-%d')
            adjusted_profile.append(new_item)
        
        # 创建合并后的用户数据
        combined_user = {
            "user_id": other_user['user_id'],
            "profile": adjusted_profile,
            "query": first_user.get("query",[]) + other_user.get("query", [])  # 如果有query字段也一并保留
        }
        
        # 添加到数据集列表
        mixed_dataset.append(combined_user)
        
    return mixed_dataset

def generate_drift_dataset(original_data, drift_level):
    """
    根据漂移级别生成数据集
    drift_level: 0.0 (只保留后半部分), 0.25 (前25%，后75%), 0.5 (前后各50%), 0.75 (前75%，后25%), 1.0 (只保留前半部分)
    """
    new_data = []
    
    for user in original_data:
        # 复制除profile和query外的用户数据
        user_copy = {}
        for key, value in user.items():
            if key not in ['profile', 'query']:
                user_copy[key] = value 
        
        profile = user.get('profile', [])
        query = user.get('query', [])
        
        if not profile:
            user_copy['profile'] = []
            user_copy['query'] = query  # 保留原始query
            new_data.append(user_copy)
            continue
        
        if not query:
            user_copy['query'] = []
            user_copy['profile'] = profile  # 保留原始profile
            new_data.append(user_copy)
            continue
            
        # 找到分割点
        profile_split_point = find_profile_split_point(profile)
        query_split_point = find_query_split_point(query)
        
        # 分割profile为两部分（用户A和用户B）
        profile_A = profile[:profile_split_point]
        profile_B = profile[profile_split_point:]
        
        # 分割query为两部分（用户A和用户B）
        query_A = query[:query_split_point]
        query_B = query[query_split_point:]
        
        # 根据漂移级别组合profile数据
        if drift_level == 0.0:
            # 只保留用户B的数据（后半部分）
            combined_profile = profile_B[:]
        elif drift_level == 1.0:
            combined_profile = profile_A[:]
        elif drift_level == 0.25 or drift_level == 0.5:
            total_desired_size = len(profile_B)/(1-drift_level)
            profile_A_size = int(total_desired_size * drift_level)
            profile_B_size = len(profile_B)
            profile_A_selected = profile_A[-profile_A_size:] if len(profile_A) >= profile_A_size else profile_A
            profile_B_selected = profile_B[:profile_B_size] if len(profile_B) >= profile_B_size else profile_B
            combined_profile = profile_A_selected + profile_B_selected
        elif drift_level ==0.75:
            total_desired_size = int(len(profile_A)/drift_level)
            profile_A_size = len(profile_A)
            profile_B_size = total_desired_size - profile_A_size
            # 从用户A取后profile_A_size条，从用户B取前profile_B_size条
            profile_A_selected = profile_A
            profile_B_selected = profile_B[:profile_B_size]
            combined_profile = profile_A_selected + profile_B_selected
        
        # 调整时间戳以确保时间连续性
        if drift_level != 0.0 and drift_level != 1.0:
            # 只有当混合了两个部分时才调整时间戳
            try:
                profile_A_for_ts, profile_B_for_ts = adjust_timestamps(
                    profile_A_selected if 'profile_A_selected' in locals() else profile_A,
                    profile_B_selected if 'profile_B_selected' in locals() else profile_B
                )
                # 重新组合已调整时间戳的profile
                combined_profile = profile_A_for_ts + profile_B_for_ts
            except:
                pass  # 如果时间戳调整失败，使用未调整的combined_profile
        
        user_copy['profile'] = combined_profile
        # query只使用用户B的数据
        user_copy['query'] = query_B
        
        new_data.append(user_copy)
    
    return new_data

def generate_test_dataset(dataset,task_id):
    """
    从数据集中提取查询，生成测试数据集
    """
    test_dataset = []
    for user in dataset:
        for item in user['query']:
            test_item = {
                "id": item['id'],
                "output": item['gold']
            }
            test_dataset.append(test_item)

    final_test_dataset = {
        "task": f"LaMP_{task_id}",
        "golds": test_dataset
    }
    
    return final_test_dataset

def main(task_id):
    # 读取原始数据
    input_file, task_name = get_task_path(task_id)
    
    with open(input_file, 'r') as f:
        user_data = json.load(f)
    
    # 提取第一个用户和后续用户用于混合
    first_user = user_data[0]  # 第一个用户作为基础
    other_users = user_data[1:11]  # 后续10个用户
    
    # 创建混合数据
    mixed_dataset = mix_users_data(first_user, other_users)
    
    # 创建测试集（使用后续10个用户的query作为测试数据）
    test_dataset = []
    for other_user in other_users:
        for item in other_user['query']:
            test_item = {
                "id": item['id'],
                "output": item['gold']
            }
            test_dataset.append(test_item)

    final_test_dataset = {
        "task": f"LaMP_{task_id}",
        "golds": test_dataset
    }
    
    # 创建输出目录
    output_dir = f"./data/{task_name}/drift"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存混合数据集
    mixed_output_file = os.path.join(output_dir, 'drift_train.json')
    with open(mixed_output_file, 'w') as f:
        json.dump(mixed_dataset, f, indent=2)
    
    # 保存测试集
    test_output_file = os.path.join(output_dir, 'drift_test.json')
    with open(test_output_file, 'w') as f:
        json.dump(final_test_dataset, f, indent=2)
    
    print("混合训练集和测试集已生成！")
    print(f"训练集包含 {len(mixed_dataset)} 个用户数据")
    print(f"测试集包含 {len(test_dataset)} 个用户查询")
    
    # 生成不同漂移级别的数据集
    drift_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # 使用混合后的数据作为原始数据生成漂移数据集
    for drift_level in drift_levels:
        new_dataset = generate_drift_dataset(mixed_dataset, drift_level)
        
        # 保存新数据集
        output_file = os.path.join(output_dir, f'drift_train_{int(drift_level*100)}.json')
        with open(output_file, 'w') as f:
            json.dump(new_dataset, f, indent=2)
        
        print(f"已生成漂移级别 {drift_level} 的数据集，保存至 {output_file}")
        
        # 生成对应的测试数据集
        test_dataset = generate_test_dataset(new_dataset, task_id)
        test_output_file = os.path.join(output_dir, f'drift_test_{int(drift_level*100)}.json')
        with open(test_output_file, 'w') as f:
            json.dump(test_dataset, f, indent=2)
        
        print(f"已生成漂移级别 {drift_level} 的测试数据集，保存至 {test_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LaMP任务数据漂移生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python script.py --task_id 2M
  python script.py -t 2N
  
任务编号对应关系:
  1: citation (引用任务)
  2M: movie_tagging (电影标记任务)
  2N: news_categorize (新闻分类任务)
  4: news_headline (新闻标题任务)
  5: product_rating (产品评分任务)
  6: scholarly_title (学术标题任务)
  7: tweet_paraphrase (推文改写任务)
        """
    )
    
    # 添加--task_id参数，支持中英混写的键名
    parser.add_argument(
        "--task_id", "-t",
        type=str,
        required=True,
        choices=["1", "2M", "2N", "3", "4", "5", "7"],  # 修改为字符串选择
    )
    
    # 添加可选参数：混合用户数量
    parser.add_argument(
        "--mix_users", "-m",
        type=int,
        default=10,
        help="用于混合的用户数量（默认为10）"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数
    main(args.task_id)