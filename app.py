import os
import json
import uuid
import datetime
import streamlit as st
import numpy as np
import torch
from typing import Dict, List, Any, TypedDict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationBufferMemory


# 상태 정의
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    user_id: str
    session_id: str
    context: Dict[str, Any]
    system_prompt: str
    current_response: str


# 스레드 관리
class ThreadManager:
    def __init__(self, checkpoint_dir="chat_threads"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._thread_states = {}
        self._load_thread_states()
    
    def _load_thread_states(self):
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                thread_id = filename[:-5]
                file_path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._thread_states[thread_id] = json.load(f)
                except Exception as e:
                    print(f"스레드 로드 오류: {e}")
    
    def _save_thread_state(self, thread_id: str, state: ChatState):
        file_path = os.path.join(self.checkpoint_dir, f"{thread_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def create_thread(self, user_id: str, system_prompt: str) -> str:
        thread_id = str(uuid.uuid4())
        initial_state = ChatState(
            messages=[{"role": "system", "content": system_prompt}],
            user_id=user_id,
            session_id=thread_id,
            context={},
            system_prompt=system_prompt,
            current_response=""
        )
        self._thread_states[thread_id] = initial_state
        self._save_thread_state(thread_id, initial_state)
        return thread_id
    
    def get_thread(self, thread_id: str) -> Optional[ChatState]:
        if thread_id not in self._thread_states:
            file_path = os.path.join(self.checkpoint_dir, f"{thread_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._thread_states[thread_id] = json.load(f)
                    return self._thread_states[thread_id]
            return None
        return self._thread_states.get(thread_id)
    
    def update_thread(self, thread_id: str, state: ChatState) -> None:
        self._thread_states[thread_id] = state
        self._save_thread_state(thread_id, state)
    
    def list_threads(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        threads = []
        for thread_id, state in self._thread_states.items():
            if user_id is None or state["user_id"] == user_id:
                title = "무제 대화"
                for msg in state.get("messages", []):
                    if msg.get("role") == "user":
                        title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                        break
                
                threads.append({
                    "thread_id": thread_id,
                    "user_id": state.get("user_id"),
                    "title": title,
                    "last_updated": datetime.datetime.now().isoformat(),
                    "message_count": len(state.get("messages", []))
                })
        return sorted(threads, key=lambda x: x["last_updated"], reverse=True)


# 컨텍스트 검색
class ContextRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.user_context_dir = "user_context"
        os.makedirs(self.user_context_dir, exist_ok=True)
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        file_path = os.path.join(self.user_context_dir, f"{user_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def find_similar_messages(self, thread_manager, query, user_id, current_thread_id, top_k=3):
        query_embedding = self.embedding_model.encode(query)
        all_threads = thread_manager.list_threads(user_id)
        similar_messages = []
        
        for thread_info in all_threads:
            thread_id = thread_info["thread_id"]
            if thread_id == current_thread_id:
                continue
                
            thread_state = thread_manager.get_thread(thread_id)
            if not thread_state:
                continue
            
            for i, msg in enumerate(thread_state["messages"]):
                if msg["role"] != "user":
                    continue
                    
                try:
                    message_embedding = self.embedding_model.encode(msg["content"])
                    similarity = np.dot(query_embedding, message_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(message_embedding)
                    )
                    
                    response = ""
                    if i + 1 < len(thread_state["messages"]) and thread_state["messages"][i + 1]["role"] == "assistant":
                        response = thread_state["messages"][i + 1]["content"]
                    
                    similar_messages.append({
                        "thread_id": thread_id,
                        "content": msg["content"],
                        "response": response,
                        "similarity": float(similarity)
                    })
                except Exception:
                    continue
        
        similar_messages.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_messages[:top_k]


# 스트리밍 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.new_tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_tokens.append(token)
        self.container.markdown(self.text)


# 모델 로딩 함수
@st.cache_resource
def load_model():
    try:
        model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"모델 로딩 오류: {e}")
        return None, None


@st.cache_resource
def load_embedding_model():
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


def load_system_prompt(file_path="./prompt/voicechat_prompt_0331.txt"):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return "당신은 도움이 되는 AI 어시스턴트입니다."


# Langgraph 노드 함수들
def retrieve_context(state: ChatState, context_retriever: ContextRetriever, thread_manager: ThreadManager) -> ChatState:
    last_user_message = next((msg["content"] for msg in reversed(state["messages"]) if msg["role"] == "user"), None)
    
    if last_user_message:
        similar_messages = context_retriever.find_similar_messages(
            thread_manager, last_user_message, state["user_id"], state["session_id"]
        )
        user_context = context_retriever.get_user_context(state["user_id"])
        state["context"] = {
            "similar_messages": similar_messages,
            "user_context": user_context
        }
    
    return state


def generate_response(state: ChatState, model, tokenizer) -> ChatState:
    try:
        enhanced_messages = state["messages"].copy()
        
        if "similar_messages" in state["context"] and state["context"]["similar_messages"]:
            similar_content = ""
            for idx, msg in enumerate(state["context"]["similar_messages"]):
                if msg["similarity"] > 0.7:
                    similar_content += f"관련 과거 대화 {idx+1}: '{msg['content']}'\n"
            
            if similar_content:
                for i, msg in enumerate(enhanced_messages):
                    if msg["role"] == "system":
                        enhanced_messages[i]["content"] = f"{state['system_prompt']}\n\n참고할 만한 정보:\n{similar_content}"
                        break
        
        text = tokenizer.apply_chat_template(
            enhanced_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            generation_args = {
                "input_ids": inputs.input_ids,
                "max_new_tokens": 128,
                "do_sample": True,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            generated_ids = model.generate(
                **generation_args,
                return_dict_in_generate=True,
                output_scores=False
            )
            
            output_ids = generated_ids.sequences[0][len(inputs.input_ids[0]):]
            complete_response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        state["current_response"] = complete_response
        state["messages"].append({"role": "assistant", "content": complete_response})
        
    except Exception as e:
        error_message = f"응답 생성 오류: {str(e)}"
        state["current_response"] = error_message
        state["messages"].append({"role": "assistant", "content": error_message})
    
    return state


# Langgraph 워크플로우 구성
def create_chat_graph(model, tokenizer, context_retriever, thread_manager) -> StateGraph:
    workflow = StateGraph(ChatState)
    
    workflow.add_node("retrieve_context", lambda state: retrieve_context(state, context_retriever, thread_manager))
    workflow.add_node("generate", lambda state: generate_response(state, model, tokenizer))
    
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# 메인 애플리케이션
def main():
    st.set_page_config(page_title="PengGo 🐧", layout="wide")
    st.title("PengGo 🐧")
    
    # 모델 및 컴포넌트 로드
    model, tokenizer = load_model()
    embedding_model = load_embedding_model()
    system_prompt = load_system_prompt()
    memory = ConversationBufferMemory(llm=model, return_messages=True)
    
    # 스레드 관리자 및 컨텍스트 검색기 초기화
    thread_manager = ThreadManager(checkpoint_dir="chat_threads")
    context_retriever = ContextRetriever(embedding_model)
    
    # Langgraph 워크플로우 생성
    chat_graph = create_chat_graph(model, tokenizer, context_retriever, thread_manager)
    
    # 세션 상태 초기화
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{str(uuid.uuid4())}"
    
    # 메모리 관리를 위한 세션 상태 초기화
    if "memory" not in st.session_state:
        st.session_state.memory = memory
        # 시스템 프롬프트를 시스템 메시지로 추가
        st.session_state.memory.chat_memory.add_message(SystemMessage(content=system_prompt))
        
    # 현재 스레드 상태 로드 또는 새 스레드 생성
    current_state = None
    if "thread_id" in st.session_state:
        current_state = thread_manager.get_thread(st.session_state.thread_id)
        
        # 현재 스레드의 모든 메시지를 메모리에 로드
        if current_state and "memory_loaded" not in st.session_state:
            # 먼저 시스템 메시지 추가
            for msg in current_state["messages"]:
                if msg["role"] == "system":
                    st.session_state.memory.chat_memory.add_message(SystemMessage(content=msg["content"]))
                    break
                    
            # 나머지 메시지 추가
            for i in range(len(current_state["messages"])):
                msg = current_state["messages"][i]
                if msg["role"] == "user":
                    st.session_state.memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    st.session_state.memory.chat_memory.add_message(AIMessage(content=msg["content"]))
                    
            st.session_state.memory_loaded = True
    
    if not current_state:
        thread_id = thread_manager.create_thread(st.session_state.user_id, system_prompt)
        st.session_state.thread_id = thread_id
        current_state = thread_manager.get_thread(thread_id)
        
        if not current_state:
            st.error("새 스레드를 생성할 수 없습니다.")
            st.stop()
    
    # 사이드바: 대화 세션 관리
    with st.sidebar:
        st.header("대화 세션 관리")
        
        if st.button("새 대화 시작"):
            # 새 스레드 생성
            thread_id = thread_manager.create_thread(
                st.session_state.user_id, system_prompt
            )
            st.session_state.thread_id = thread_id
            
            # 메모리 초기화
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            # 시스템 프롬프트 추가
            st.session_state.memory.chat_memory.add_message(SystemMessage(content=system_prompt))
            
            if "memory_loaded" in st.session_state:
                del st.session_state.memory_loaded
                
            st.rerun()
        
        # 저장된 스레드 목록
        st.subheader("저장된 대화")
        threads = thread_manager.list_threads(st.session_state.user_id)
        
        for thread in threads:
            thread_id = thread.get("thread_id")
            title = thread.get("title", "무제 대화")
            updated_at = thread.get("last_updated", "Unknown")
            
            if isinstance(updated_at, str) and len(updated_at) > 16:
                updated_at = updated_at[:16]
                
            if st.button(f"{updated_at} - {title}", key=thread_id):
                st.session_state.thread_id = thread_id
                
                # 메모리 초기화
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
                # 초기화 후 메모리 로딩을 위해 memory_loaded 플래그 삭제
                if "memory_loaded" in st.session_state:
                    del st.session_state.memory_loaded
                    
                st.rerun()
    
    # 대화 메시지 표시
    for message in current_state["messages"]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 사용자 입력 및 응답 생성
    if prompt := st.chat_input("무엇을 도와드릴까요?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        current_state["messages"].append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)
            
            result_state = chat_graph.invoke(current_state)
            st.session_state.memory.save_context({"user": prompt}, {"assistant": result_state["current_response"]})
            
            response = result_state["current_response"]
            for i in range(0, len(response), 2):
                stream_handler.on_llm_new_token(response[i:i+2])
                import time
                time.sleep(0.01)
        
        thread_manager.update_thread(st.session_state.thread_id, result_state)        
        memory_content = st.session_state.memory.load_memory_variables({})['history']
        # final_message = memory_content[2:]
        # message_contents = [msg.content for msg in history]

if __name__ == "__main__":
    main()
