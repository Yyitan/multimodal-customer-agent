# OrderedDict：有序字典，记住 key 插入 / 访问顺序，用来实现 LRU（最近最少使用）淘汰
# deque：双端队列，适合频繁头尾增删；支持maxlen，队列满了自动弹出最左边旧数据
from collections import OrderedDict,deque
# RLock：可重入锁（递归锁），多线程并发安全
from threading import RLock
from typing import TypedDict

# 定义一轮对话的结构：一次用户提问 + 一次模型回答。
class ConversationTurn(TypedDict):
    question:str
    answer:str

class ConversationStore:
    """保存有限数量的内存对话。"""

    def __init__(
        self,
        max_turns:int=6,
        max_sessions:int=1000,
    )->None:
        self.max_turns=max_turns # 单个session最多保存6轮问答
        self.max_sessions=max_sessions # 全局最多同时保存1000哥不同用户会话
  
        # _sessions:key=session_id,value=deque存多轮对话
        # _sessions：核心存储容器
        self._sessions: OrderedDict[
            str,
            deque[ConversationTurn],
        ] = OrderedDict()

        # _lock = RLock()：可重入锁;RLock：同一个线程可以多次 acquire 锁，不会死锁；普通 Lock 同一个线程拿两次锁会卡住。FastAPI 是多线程处理请求，多个请求线程同时读写_sessions，不加锁会出现数据错乱.with self._lock: 自动加锁、代码块结束自动释放锁。
        self._lock=RLock()
  
    def get_history(
        self,
        session_id:str,
    )->list[ConversationTurn]:
        """获取对话历史"""
        # with self._lock：进入方法上锁，多线程排队读写
        with self._lock:
            # 根据 session_id 拿到该用户的对话队列，不存在返回空列表
            turns=self._sessions.get(session_id)
            if turns is None:
                return []
            # 只要访问了这个会话，就把这个 session_id 移动到有序字典末尾，标记为「最近刚用过」
            self._sessions.move_to_end(session_id)
            # 返回拷贝，防止外部代码直接修改内部存储的原始对话数据，避免外部篡改污染内部状态。
            return [turn.copy() for turn in turns]

    def append_turn(
        self,
        session_id:str,
        question:str,
        answer:str,
    )->None:
        """新增一轮问答（核心写入方法）"""
        clean_question=question.strip()
        clean_answer=answer.strip()
    
        if not clean_question or not clean_answer:
            return
 
        with self._lock:
            # 这个session不存在
            if session_id not in self._sessions:
                # 判断全局对话数量是否超限
                if len(self._sessions)>=self.max_sessions:
                    # popitem(last=False) 弹出有序字典最前面的key，也就是最久没访问的session
                    self._sessions.popitem(last=False)
     
                # 创建新会话队列，deque设置maxlen=self.max_turns=6
                # deque自带特性：队列元素达到maxlen之后，新增元素会自动删掉最左边旧的一轮对话
                self._sessions[session_id]=deque(maxlen=self.max_turns) 

            self._sessions[session_id].append({
                "question":clean_question,
                "answer":clean_answer,
            })   

            self._sessions.move_to_end(session_id)
 
    def clear(self,session_id:str)->bool:
        """clear 删除指定会话"""
        with self._lock:
            # pop：存在就删掉，返回被删value；不存在返回None
            return self._sessions.pop(session_id,None) is not None

def format_conversation_history(
    history:list[ConversationTurn],
)->str:
    """格式化函数"""
    lines:list[str]=[]
    for turn in history:
        # 从get_history()拿到对话列表
        question=turn.get("question","").strip()
        answer = turn.get("answer", "").strip()
        # 输出：拼接成一段文本字符串，示例：
        if question:
            lines.append(f"用户：{question}")
        if answer:
            lines.append(f"客服：{answer}")
    return "\n".join(lines)