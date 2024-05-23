import re
from datetime import datetime
import os
from collections import defaultdict

def get_time_password():
    now = datetime.now()
    # 生成时间密码，四位数字，如：2020年12月31日 05:30 -> 0530
    time_password = now.strftime('%H%M')
    return time_password

def beautify_paper_output(output_str):
    orgin = output_str
    paper = defaultdict(str)

    '''
    1. Title: Automated Feedback Generation for Competition-Level Code （竞赛级代码自动反馈生成）

    2. Authors: Jialu Zhang, De Li, John C. Kolesar, Hanyuan Shi, Ruzica Piskac

    3. Affiliation: 耶鲁大学

    4. Keywords: Automated Feedback, Code Repair, Competition-Level Programming, Machine Learning

    '''

    # 从字符串中提取出title, authors, keywords
    title = re.findall(r'Title: (.*)\n', output_str)[0]
    authors = re.findall(r'Authors: (.*)\n', output_str)[0]
    keywords = re.findall(r'Keywords: (.*)\n', output_str)[0]

    return title, authors, keywords