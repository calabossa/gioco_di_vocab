# -*- coding: utf-8 -*-
"""

@author: Boaz
"""

import random
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os


# ==== CONFIG ====
MODEL_NAME = "gpt-4o-mini"   
load_dotenv() 

# ==== DATA MODEL FOR LLM OUTPUT ====
class EvalResult(BaseModel):
    score: int  # 1 = wrong, 2 = partially correct, 3 = correct
    feedback: str
    correct_answer_to_show: str


# ==== INITIALISE LLM WITH STRUCTURED OUTPUT ====
@st.cache_resource
def get_structured_llm():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    # Ask for structured output 
    return llm.with_structured_output(EvalResult, method="json_mode")


# ==== LOAD VOCAB ====
@st.cache_data
def load_vocab():
    SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")
    df = pd.read_csv(SHEET_CSV_URL)
    df = df[["ITALIANO", "TRADUZIONE"]]
    df = df.rename(columns={"ITALIANO": "italian", "TRADUZIONE": "translation"})
    df = df.dropna(subset=["italian", "translation"])    
    df["italian"] = df["italian"].astype(str).str.strip()
    df["translation"] = df["translation"].astype(str).str.strip()
    
    return df


class NewWordCard(BaseModel):
    italian: str
    translation: str

def generate_new_word_card(llm, df, max_attempts=5) -> dict:
    """
    Ask the LLM to invent one new intermediate-level Italian word with translation.
    Returns a card dict with same keys as cards from the sheet.
    """
    structured_llm = llm.with_structured_output(NewWordCard, method="json_mode")

    system_prompt = """
    You are an Italian teacher.
    
    Think of one random Italian word which is suitable for an advanced learner (roughly level C1). Be creative, think of something not necessarily common.
    It can be a verb, an object, and even a short expression.
    Avoid proper names and extremely rare terms. 
    Provide:
    - the Italian word you thought of (lemma)
    - a simple translation of that word in one language (preferably English, but French is also ok).
    You MUST respond ONLY as JSON describing the word, with keys "italian" and "translation".
    Do not include any extra text outside the JSON.
    """

    for _ in range(max_attempts):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate one word now in JSON form.")
        ]
        result: NewWordCard = structured_llm.invoke(messages)

        italian_word = result.italian.strip()
        key = italian_word.lower()

        # reject if already known (in sheet or previously generated this session)
        if key not in st.session_state.seen_italian:
            st.session_state.seen_italian.add(key)
            return {
                "italian": italian_word,
                "translation": result.translation.strip(),
                "direction_it_to_other": random.choice([True, False]),
                "is_generated": True,
            }

    # fallback if we couldn't find a new word
    return pick_random_card(df)


def pick_random_card(df: pd.DataFrame):
    row = df.sample(1).iloc[0]
    italian = str(row["italian"]).strip()
    translation = str(row["translation"]).strip()

    # pick a random direction (it-other or other-it)
    direction_it_to_other = random.choice([True, False])

    return {
        "italian": italian,
        "translation": translation,
        "direction_it_to_other": direction_it_to_other,
    }


def pick_any_card(df, base_llm):
    """
    Sometimes returns a sheet card, sometimes a generated one.
    """
    # probability for a generated card (from 0 to 1)
    if random.random() < 0.1:
        return generate_new_word_card(base_llm, df)
    else:
        return pick_random_card(df)


def build_question(card: dict) -> str:
    prefix = "⭐ NEW WORD: " if card.get("is_generated") else ""
    if card["direction_it_to_other"]:
        # Show Italian, ask for translation
        return f"{prefix}Translate from Italian: **{card['italian']}**"
    else:
        # Show translation, ask for Italian
        return f"{prefix}Traduci in italiano: **{card['translation']}**"


def evaluate_answer(card: dict, student_answer: str) -> EvalResult:
    structured_llm = get_structured_llm()
    student_answer = student_answer.strip()

    if card["direction_it_to_other"]:
        # ITALIAN → OTHER 
        system_prompt = """
    You are an Italian vocabulary trainer.
    
    Mode: from_italian.
    
    You receive:
    - italian_word: the Italian word to learn.
    - translation_field: a reference translation, which may be in English, French, or Hebrew.
    - student_answer: the student's answer.
    
    The student was shown the ITALIAN word and asked to translate it.
    They are allowed to answer in ANY of these languages: English, French, or Hebrew.
    You care about MEANING, not which of those three languages they choose. 
    Articles should be ignored. For instance the translations "bed", "a bed" and "the bed" are all similar for you, the articles "a" and "the" should be ignored by you.
    
    Scoring:
    - 3 = meaning correct (or very close), in any of EN/FR/HE. Allow small spelling mistakes.
    - 2 = partially correct (approximate meaning, or minor issue).
    - 1 = incorrect meaning.
    
    feedback:
    - One short sentence in English.
    
    correct_answer_to_show:
    - A good translation in EN/FR/HE, usually based on translation_field, possibly with a short alternative
      (e.g. "blind (FR: aveugle)").
    """

        payload = {
            "italian_word": card["italian"],
            "translation_field": card["translation"],
            "student_answer": student_answer,
        }

    else:
        # OTHER → ITALIAN 
        system_prompt = """
    You are an Italian vocabulary trainer.
    
    Mode: to_italian.
    
    You receive:
    - italian_word: the correct Italian word.
    - translation_field: a cue word or phrase in English, French, or Hebrew shown to the student.
    - student_answer: the student's answer.
    
    The student was shown translation_field (EN/FR/HE) and must answer in ITALIAN.
    So the ONLY correct target language is Italian.
    
    Scoring:
    - 3 = meaning correct (allow small spelling mistakes, gender issues). Articles are not important and you should ignore them. For instance the translations "letto", "un letto" and "il letto" are all similar for you, the articles "un" and "il" should be ignored by you.
    - 2 = partially correct (roughly right meaning but form is flawed or meaning is from the same general field but not truly correct).
    - 1 = incorrect, including cases where the meaning is OK but the answer is not in Italian.
    
    feedback:
    - One short sentence in English.
    
    correct_answer_to_show:
    - Always the best clean Italian form, normally equal or very close to italian_word.
    - Never just repeat translation_field as the correct answer.
    """

        payload = {
            "italian_word": card["italian"],
            "translation_field": card["translation"],
            "student_answer": student_answer,
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the data as JSON:\n{payload}"),
    ]

    result: EvalResult = structured_llm.invoke(messages)
    return result

# ==== call STREAMLIT ====
def main():
    st.set_page_config(page_title="Italian Vocab Trainer", page_icon="🇮🇹")
    st.title("🇮🇹 Gioco di vocab")
    
    # End game button
    if st.button("🛑 End game"):
        st.write("Game ended. You can close this tab.")
        os._exit(0)


    df = load_vocab()
    # get all words i list for comparing with "new" 
    if "seen_italian" not in st.session_state:
        st.session_state.seen_italian = {w.strip().lower() for w in df["italian"]}

    if "card" not in st.session_state:
        st.session_state.card = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "score_total" not in st.session_state:
        st.session_state.score_total = 0
    if "rounds" not in st.session_state:
        st.session_state.rounds = 0

    if "generated_italian" not in st.session_state:
        st.session_state.generated_italian = set()
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 New word"):
            base_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)  # or reuse your existing getter
            st.session_state.card = pick_any_card(df, base_llm)
            st.session_state.last_result = None
            st.session_state.student_answer = ""

    with col2:
        st.write(f"Rounds: {st.session_state.rounds} | Score: {st.session_state.score_total}")

    if st.session_state.card is None:
        st.info("Click **New word** to start!")
        return

    # Show the question
    question_text = build_question(st.session_state.card)
    st.markdown(question_text)

    # Input for the answer
    answer = st.text_input("Your answer:", key="student_answer")

    if st.button("✅ Check answer"):
        if not answer.strip():
            st.warning("Please type an answer first.")
        else:
            result = evaluate_answer(st.session_state.card, answer.strip())
            st.session_state.last_result = result
            st.session_state.rounds += 1
            if result.score == 3:
                st.session_state.score_total += 2
            elif result.score == 2:
                st.session_state.score_total += 1

    # Show feedback
    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        if res.score == 3:
            st.success(f"✅ Correct! {res.feedback}")
        elif res.score == 2:
            st.warning(f"⚠️ Almost: {res.feedback}")
        else:
            st.error(f"❌ Not quite: {res.feedback}")

        st.write("**Expected answer:** ", res.correct_answer_to_show)


if __name__ == "__main__":
    main()
