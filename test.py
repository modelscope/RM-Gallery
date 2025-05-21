
import re


def extract_tags_content(s):
    pattern = r'<([^>]+)>(.*?)</\1>'
    matches = re.findall(pattern, s)
    
    # 提取每个匹配中的内容部分
    contents = [match[1] for match in matches]
    return contents



# 示例字符串
string = "<abc>content1</abc><xyz>content2</xyz><def>content3</def>"

# 提取内容
contents = extract_tags_content(string)

# 输出结果
print(contents)