"""Writer LLM 客户端:用本地 vLLM(OpenAI 兼容)扩充剧本场景与事件措辞。

设计原则(与 MPFD paraphrase 配方一致):
  - LLM 只提供语言多样性;类别/标签由代码与校验器强制,违规样本直接丢弃;
  - 语言标签(yue/en/mixed)由代码从字符组成推断,不信任模型自报;
  - 多样性:同前缀(前5字符)去重,防模板枚举("and next X"病)。

服务器用法(先 bash serve_qwen.sh 起服务):
  python writer_llm.py --task scenarios --n 100 --out ../../data/synth/scenarios.json
  python writer_llm.py --task wordings  --n 60  --out ../../data/synth/wordings.json
本地无 GPU 冒烟:加 --mock(用内置样例响应走通全流程)。
产物拷回本地后:gen_v0.py --scenarios ... --wordings ... 直接消费。

仅用 stdlib(urllib),服务器任何 env 可跑。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synth.script import SLOT_BANK  # noqa: E402

CJK = re.compile(r"[一-鿿]")
ASCII_ALPHA = re.compile(r"[A-Za-z]")

# ---------------- prompts ----------------

PROMPT_SCENARIOS = """你在为香港电讯客服的语音对话数据集写剧本。生成 {k} 个不同的客服场景,每个包含:
- scenario: 英文短id(小写下划线)
- user_q: 用户的粤语口语问题(地道口语,可自然夹英文词,6-40字)
- agent_answer: 客服的粤语长回答(30-90字,信息量足,像真人客服连续讲解)

硬性要求:
1. agent_answer 必须包含 1-2 个槽位占位符,只能用这些: {slots}
2. user_q 里的占位符(如有)必须也出现在 agent_answer 里
3. 全部用地道香港粤语口语字(係/嘅/咗/唔/畀/嗰),不要书面语普通话
4. 场景要多样:账单/套餐/维修/漫游/退款/宽带/号码携转/合约/增值服务等
5. 每个场景的 agent_answer 开头用词必须不同

只输出 JSON 数组,不要任何其他文字:
[{{"scenario":"...","user_q":"...","agent_answer":"..."}}]"""

PROMPT_WORDINGS = """你在为粤英混说语音对话数据集扩充"用户插话"措辞库。生成第 {batch} 批,每类 {k} 条,五类:

1. backchannel(应声,表示在听,不打断): 极短,如"係啊/嗯嗯/okok",允许粤语、英语、粤英混
2. filler(填充词/迟疑): 如"即係…/erm"
3. side_speech(用户转头对屋企人讲的话,与通话无关): 如"阿媽幫我攞杯水"
4. floor_claim(礼貌打断,想插入新问题): 如"唔好意思打斷一下…"
5. correction_pattern(纠正模板,必须含 {{new}} 占位符,表示正确的值): 如"唔係啊,我問嘅係{{new}}"

硬性要求:
- 地道香港粤语口语字;correction_pattern 中约一半要粤英句内混说(如"no no 係{{new}}先啱")
- correction_pattern 只能含 {{new}} 这一个占位符
- 同类内措辞开头不能重复

只输出 JSON 对象,不要任何其他文字:
{{"backchannel":["..."],"filler":["..."],"side_speech":["..."],"floor_claim":["..."],"correction_pattern":["..."]}}"""

# ---------------- 校验器(类别不变量,LLM 产物的准入门) ----------------


def lang_of(text: str) -> str:
    has_cjk = bool(CJK.search(text))
    has_lat = bool(ASCII_ALPHA.search(text))
    if has_cjk and has_lat:
        return "mixed"
    return "yue" if has_cjk else "en"


def validate_scenario(item: dict) -> str | None:
    """返回 None=通过,否则拒绝原因。"""
    try:
        sid, q, a = item["scenario"], item["user_q"], item["agent_answer"]
    except (KeyError, TypeError):
        return "missing keys"
    if not re.fullmatch(r"[a-z0-9_]{3,30}", str(sid)):
        return "bad scenario id"
    ph_q = set(re.findall(r"\{(\w+)\}", q))
    ph_a = set(re.findall(r"\{(\w+)\}", a))
    if not ph_a:
        return "answer has no slot"
    if not (ph_q | ph_a) <= set(SLOT_BANK):
        return f"unknown slot {sorted((ph_q | ph_a) - set(SLOT_BANK))}"
    if not ph_q <= ph_a:
        return "q slot not in answer"
    plain_a = re.sub(r"\{\w+\}", "", a)
    if not (20 <= len(plain_a) <= 100):
        return "answer length"
    if not (4 <= len(re.sub(r"\{\w+\}", "", q)) <= 45):
        return "q length"
    if "\n" in q or "\n" in a:
        return "newline"
    if not CJK.search(a):
        return "answer not cantonese"
    for s in (q, a):  # 散落花括号会让 format() 崩
        if s.count("{") != len(re.findall(r"\{\w+\}", s)) or s.count("{") != s.count("}"):
            return "stray brace"
    return None


WORDING_RULES = {
    "backchannel": (1, 6, False),
    "filler": (1, 8, False),
    "side_speech": (4, 22, False),
    "floor_claim": (5, 32, False),
    "correction_pattern": (3, 32, True),  # 必须含 {new};下限按去掉{new}后的字数
}


def validate_wording(cls: str, text: str) -> str | None:
    lo, hi, need_new = WORDING_RULES[cls]
    plain = text.replace("{new}", "")
    if not (lo <= len(plain.replace(" ", "")) <= hi):
        return "length"
    ph = set(re.findall(r"\{(\w+)\}", text))
    if need_new:
        if ph != {"new"}:
            return "placeholder must be exactly {new}"
    elif ph:
        return "unexpected placeholder"
    if "\n" in text:
        return "newline"
    return None


# ---------------- vLLM 客户端 ----------------


_THINKING_KWARG_OK = True  # 无 thinking 模板(如 Instruct-2507)不接受该参数时自动降级


def chat(base_url: str, model: str, prompt: str, temperature: float,
         timeout: int = 300) -> str:
    global _THINKING_KWARG_OK
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if _THINKING_KWARG_OK:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if _THINKING_KWARG_OK and e.code == 400:
            _THINKING_KWARG_OK = False
            print("  (server rejected enable_thinking kwarg, retrying without)")
            return chat(base_url, model, prompt, temperature, timeout)
        raise
    return data["choices"][0]["message"]["content"]


def extract_json(text: str):
    """截取首个 JSON 数组/对象(容忍模型加```包裹或前后废话)。"""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)
    candidates = sorted(
        [(text.find(o), o, c) for o, c in (("[", "]"), ("{", "}"))
         if text.find(o) >= 0]
    )
    for i, opener, closer in candidates:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no valid JSON found")


# ---------------- mock(本地无 GPU 冒烟) ----------------

MOCK_SCENARIOS = json.dumps([
    {"scenario": "broadband_fix", "user_q": "屋企上網成日斷線,可以點搞",
     "agent_answer": "同你檢查過條線,建議約{day}{time}師傅上門檢查個貓同埋條線,期間你可以用返手機數據頂住先"},
    {"scenario": "port_in", "user_q": "我想轉台但係想keep返個號碼",
     "agent_answer": "攜號轉台冇問題,你交返轉台紙之後{month}就會生效,期間號碼照用,唔會斷到你服務"},
])
MOCK_WORDINGS = json.dumps({
    "backchannel": ["係囉", "uh huh", "好呀"],
    "filler": ["咁即係", "hmm"],
    "side_speech": ["細佬幫我開一開門", "隻貓唔好上枱"],
    "floor_claim": ["唔好意思,插一句先", "sorry 等等,我有嘢想問"],
    "correction_pattern": ["唔係{new}咩?", "no no 係{new}先啱", "你講錯咗,係{new}"],
})


# ---------------- 主流程:生成-校验-去重-重试 ----------------


def run_scenarios(args) -> None:
    got: list[dict] = []
    seen: set = set()
    rounds = 0
    while len(got) < args.n and rounds < args.max_rounds:
        rounds += 1
        prompt = PROMPT_SCENARIOS.format(k=min(20, args.n), slots="{" + "} {".join(SLOT_BANK) + "}")
        raw = MOCK_SCENARIOS if args.mock else chat(args.base_url, args.model, prompt, args.temperature)
        try:
            items = extract_json(raw)
        except ValueError as e:
            print(f"round {rounds}: {e}")
            continue
        for it in items:
            why = validate_scenario(it)
            if why:
                print(f"  reject [{why}]: {str(it)[:60]}")
                continue
            key = re.sub(r"\{\w+\}", "", it["agent_answer"])[:5]
            if key in seen:
                print(f"  dup prefix: {key}")
                continue
            seen.add(key)
            got.append({"scenario": it["scenario"], "user_q": it["user_q"],
                        "agent_answer": it["agent_answer"]})
        print(f"round {rounds}: total {len(got)}/{args.n}")
        if args.mock:
            break
        time.sleep(0.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(got, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"-> {args.out} ({len(got)} scenarios)")


def run_wordings(args) -> None:
    banks: dict = {c: [] for c in WORDING_RULES}
    seen: dict = {c: set() for c in WORDING_RULES}
    rounds = 0
    while rounds < args.max_rounds and min(len(v) for v in banks.values()) < args.n:
        rounds += 1
        prompt = PROMPT_WORDINGS.format(batch=rounds, k=min(15, args.n))
        raw = MOCK_WORDINGS if args.mock else chat(args.base_url, args.model, prompt, args.temperature)
        try:
            obj = extract_json(raw)
        except ValueError as e:
            print(f"round {rounds}: {e}")
            continue
        for cls in WORDING_RULES:
            for text in obj.get(cls, []):
                if not isinstance(text, str):
                    continue
                why = validate_wording(cls, text)
                if why:
                    print(f"  reject {cls} [{why}]: {text[:30]}")
                    continue
                key = text.replace("{new}", "")[:4]
                if key in seen[cls]:
                    continue
                seen[cls].add(key)
                banks[cls].append([text, lang_of(text)])
        print(f"round {rounds}:", {c: len(v) for c, v in banks.items()})
        if args.mock:
            break
        time.sleep(0.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(banks, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["scenarios", "wordings"], required=True)
    ap.add_argument("--n", type=int, default=60, help="目标条数(scenarios 总数 / wordings 每类)")
    ap.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_rounds", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true", help="无服务冒烟(内置样例)")
    args = ap.parse_args()
    if args.task == "scenarios":
        run_scenarios(args)
    else:
        run_wordings(args)


if __name__ == "__main__":
    main()
