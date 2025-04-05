import json
import uuid

import mysql.connector
from typing import List, Dict
from urllib.parse import quote


def get_mysql_connection():
    """获取MySQL连接"""
    try:
        connection = mysql.connector.connect(
            host='gaea.test.mysql03.b2c.srv',  # 数据库主机地址
            port=13616,
            user='micar_site_sv22_wn',  # 数据库用户名
            password='e3vpM8dEtctXFMgKqt-eneLzK_iOoyY9',  # 数据库密码
            database='micar_site_staging'  # 数据库名称
        )
        return connection
    except mysql.connector.Error as e:
        print(f"连接MySQL失败: {e}")
        return None


connection = get_mysql_connection()


def execute_query(query: str) -> List[Dict]:
    """执行SQL查询并返回结果列表"""
    results = []
    # connection = get_mysql_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as e:
            print(f"执行查询失败: {e}")
        # finally:
        #     connection.close()

    return results


def url_encode(url: str) -> str:
    """对URL进行编码"""
    return 'https://www.xiaomiev.com/communityWeb?_rt=ft&url=' + quote(url)


def get_module_list(page_id: str) -> List[str]:
    """获取指定页面的模块列表"""
    query = f"select module_id from site_page_module_tmp where page_id='{page_id}' order by priority"
    result_list = execute_query(query)
    return [row['module_id'] for row in result_list]


def get_module_items(page_id: str, module_id: str) -> List[Dict]:
    """获取指定页面和模块的内容项列表"""
    query = f"select * from site_page_module_item_tmp where page_id='{page_id}' and module_id='{module_id}' and parent_key='' order by priority"
    result_list = execute_query(query)
    return result_list


def get_module_items_by_parent(page_id: str, module_id: str, parent_key: str) -> List[Dict]:
    """获取指定页面和模块的内容项列表"""
    query = f"select * from site_page_module_item_tmp where page_id='{page_id}' and module_id='{module_id}' and parent_key='{parent_key}' order by priority"
    result_list = execute_query(query)
    return result_list


def parse_img(item: Dict) -> Dict:
    """解析图片类型的内容项"""
    img = {
        "src": item['src'],
        "height": item['height'],
        "width": item['width']
    }
    img_fold = {
        "src": item['src_fold'],
        "height": item['height'],
        "width": item['width']
    }
    action = None
    if item['action_type'] == 'link':
        link_url = item['link_url']
        if item['link_type'] == 0:
            link_url = url_encode(link_url)

        action = {
            "type": item['action_type'],
            "linkUrl": link_url
        }
        if item['action_text'] != '':
            action['text'] = item['action_text']

    res = {
        "type": 'image',
        "image": img,
        "imageFold": img_fold,
        "action": action
    }

    if item['title'] != '':
        res['title'] = item['title']
    if item['subtitle'] != '':
        res['subTitle'] = item['subtitle']

    uuid1 = str(uuid.uuid4()).replace('-', '')
    res['_track'] = {
        'itemId': uuid1,
        'itemName': item['title']
    }

    return res


def parse_video(item: Dict) -> Dict:
    video = {
        "src": item['src'],
        'cover': item['cover'],
        "height": int(item['height']),
        "width": int(item['width']),
        'duration': int(item['duration'])
    }
    video_fold = {
        "src": item['src_fold'],
        'cover': item['cover_fold'],
        "height": int(item['height']),
        "width": int(item['width']),
        'duration': int(item['duration'])
    }
    action = None
    if item['action_type'] == 'link':

        link_url = item['link_url']
        if item['link_type'] == 0:
            link_url = url_encode(link_url) + '&transition_background_color=%23000000'

        action = {
            "type": item['action_type'],
            "linkUrl": link_url
        }

    res = {
        "type": 'video',
        "video": video,
        "videoFold": video_fold,
        "image": video,
        "imageFold": video_fold,
        "action": action
    }

    if item['title'] != '':
        res['title'] = item['title']
    if item['subtitle'] != '':
        res['subTitle'] = item['subtitle']

    uuid1 = str(uuid.uuid4()).replace('-', '')
    res['_track'] = {
        'itemId': uuid1,
        'itemName': item['title']
    }

    return res


def parse_tab_item(page_id, module_id, item: Dict) -> Dict:
    tab_item = {
        "tabId": item['key'],
        "text": item['text'],
        "selected": True if item['selected'] == 1 else False
    }
    if item['key'] == '':
        return tab_item
    items = get_module_items_by_parent(page_id, module_id, item['key'])
    if len(items) > 0:
        tab_item_content = []
        for sub_item in items:
            item = {}
            sub_type = sub_item['type']
            if sub_type == 'image':
                item = parse_img(sub_item)
            elif sub_type == 'video':
                item = parse_video(sub_item)
            tab_item_content.append(item)

        if len(tab_item_content) > 0:
            tab_item['content'] = {
                'data': tab_item_content
            }

    return tab_item


def parse_module_config(page_id, module: Dict) -> Dict:
    component = module['component']
    module_id = module['module_id']
    items = get_module_items(page_id, module_id)
    data_list = []
    # 遍历并打印每个模块项
    for item in items:
        item_type = item['type']
        obj = {}
        if item_type == 'image':
            obj = parse_img(item)
            data_list.append(obj)
        elif item_type == 'video':
            obj = parse_video(item)
            data_list.append(obj)
        elif item_type == 'tabItem':
            obj = parse_tab_item(page_id, module_id, item)
            data_list.append(obj)

    extra = None
    if module['extra'] is not None:
        extra = json.loads(module['extra'])

    if component == 'Tabs':
        return {
            "items": data_list
        }

    if component == 'Banner':
        return {
            "data": data_list
        }

    if component == 'Navigation':
        return {
            'title': module['title'],
            'columns': int(module['columns']),
            "data": data_list,
            'extra': extra
        }

    if component == 'News':
        return {
            'title': module['title'],
            "data": data_list,
            'extra': extra
        }

    if component == 'TabNews':
        return {
            'title': module['title'],
            "items": data_list
        }

    if component == 'ActivityList':
        return {
            'title': module['title'],
            "more": extra
        }

    if component == 'TabFeeds':
        return {
            'title': module['title'],
            "items": data_list
        }

    return {}


def get_page_config(page_id, page_title):
    modules = execute_query(f'select * from site_page_module_tmp where page_id = \'{page_id}\' order by priority ')
    module_config_list = []
    module_ids = []
    for module in modules:
        config = parse_module_config(page_id, module)

        track = {
            'sectionId': str(uuid.uuid4()).replace('-', ''),
            'sectionName': module['module_name']
        }
        config['_track'] = track

        print('---------------------------')
        print(module['module_id'])
        print(json.dumps(config, ensure_ascii=False))
        print('---------------------------')

        module_ids.append(module['module_id'])
        m = {
            'id': module['module_id'],
            'name': module['module_name'],
            'component': module['component'],
            'config': json.dumps(config, ensure_ascii=False)
        }

        if module['data_provider'] != '':
            m['dataProvider'] = module['data_provider']

        module_config_list.append(m)

    result = {
        'title': page_title,
        'components': module_config_list,
        "groupId": "1",
        "groupConfig": [
            {
                "list": module_ids,
                "name": "全部用户",
                "strategy": "all",
                "default": True
            }
        ]
    }

    return result


# 使用示例
if __name__ == "__main__":
    explore_page_config = get_page_config('explore', '探索页面')
    activity_page_config = get_page_config('activity', '活动页面')

    print('##################')
    print(json.dumps(explore_page_config, ensure_ascii=False))
    print(json.dumps(activity_page_config, ensure_ascii=False))
