from rm_gallery.pipeline.node import PaletteBlock


class BaseParser(PaletteBlock):
    ...


# 服务prompt，解析llm输出
class ResponseParser(BaseParser): 
    ...


# 中间服务，预处理任务等等
class TaskParser(BaseParser):
    ...




