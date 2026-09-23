# CantoDuplex

粤英双语 duplex 交互决策控制器(sidecar)。完整方案见 `PLAN.md`,种子数据采集协议见 `docs/COLLECTION_PROTOCOL.md`。

## 目录

```
PLAN.md                     # 完整方案 v1.0(§9 含已定决策)
docs/COLLECTION_PROTOCOL.md # 种子真集采集协议 v0.1(试录 2 对后修订)
harness/                    # 评测 harness(评测即产品:bench 论文 / 自家回归 / 客户验收)
  schema.py                 # Session/ProbeEvent/Action;事件类别→期望动作族映射
  metrics.py                # 误停率 / 漏停率 / 反应延迟 p50/p90(PLAN §4)
  probes.py                 # 种子探针库 + 受控位置注入计划(时间轴层)
tests/test_metrics.py       # 回归:oracle 全对 / naive-VAD 全误停 / 哑巴全漏停
scripts/crawl/
  fetch_rthk.py             # RTHK 烽煙节目(千禧年代/自由風自由PHONE):rss 或按日期 probe 12 个月存档
  fetch_youtube.py          # YouTube 粤语对话(访谈/清谈/phone-in),yt_sources.txt 配源
  yt-dlp.exe                # 独立版(本机 py3.9 装不了新版 yt-dlp)
scripts/diar/
  run_diar.py               # pyannote 3.1 批量 diarization(env cduplex,断点续跑)
  timing_stats.py           # RTTM → 双域 timing 先验(转移/插话分离,rthk vs youtube)
synth/                      # 合成流水线 ②③(纯代码,标签由构造保证)
  priors.py                 # 消费 timing_stats.json,按域采样时间量
  script.py                 # 场景剧本模板(类型化槽位;Writer LLM 接入点)
  inject.py                 # 事件注入器(类别→标签由代码定,措辞过不变量校验器)
  direct.py                 # Director:编时间轴 + 金标签 + 槽位翻转出 RESUME 三重标签
  labels.py                 # 因果 chunk 标签(80ms;ATTENUATE→CUT 相位,禁未来泄漏)
scripts/synth/gen_v0.py     # 生成入口(带 oracle 自检:误停/漏停必须全零)
scripts/synth/serve_qwen.sh # 服务器端起 vLLM(OpenAI 兼容,served-model-name=qwen)
scripts/synth/writer_llm.py # Writer 客户端:场景/措辞扩充,生成-校验-去重-重试;--mock 可离线冒烟
tests/test_synth.py         # 合成回归:纠正不变量/RESUME 自洽/截断/oracle 全零/因果相位
data/web/                   # 爬取产物(混合通道:timing 先验/编码器预训练/词表挖掘,不做控制监督;版权仅内部研究)
data/diar/timing_stats.json # 双域 timing 先验(rthk 客服域 / youtube 闲谈域)
data/synth/v0_*             # 合成时间轴 + 金标签 + chunk 标签
```

## 爬取用法(内地网络需走本地代理,默认 http://127.0.0.1:7897)

```
python scripts/crawl/fetch_rthk.py --mode probe --days 365    # RTHK 全年存档(每日 YYYYMMDD_段号.m4a 直链)
python scripts/crawl/fetch_youtube.py --limit 10              # 每源 10 条;编辑 yt_sources.txt 加源
```
manifest.jsonl / info.json 记录每条的来源、日期与 license 备注;archive.txt 支持断点续爬。

## 快速验证

```
python -m pytest tests -q
```

## 指标口径

- **误停率**:continue 族探针(应声/旁语/噪声/填充/回声)在 onset 后 2s 反应窗内被 STOP(CUT/PAUSE/YIELD)响应的比例;
- **漏停率**:stop 族探针(纠正/抢话)在 onset+0.8s 截止前没有 STOP 的比例;
- **反应延迟**:stop 族命中的首个 STOP 相对 onset 的延迟,报 p50/p90;
- 协作接话(cooperative_completion)不计两轴,单独报停止比例;
- 只统计 agent 说话期间的探针(duplex 定义),其余计 skipped。

2D 头条图:误停率(过度反应轴)× 漏停率(欠反应轴),naive-VAD 落高误停角、哑巴系统落高漏停角,好的控制器逼近原点。

## Writer LLM 服务器工作流

```
# 服务器(4 卡起服务;flashinfer 坏则 pip uninstall flashinfer)
bash scripts/synth/serve_qwen.sh /path/to/Qwen模型 8002 6,7,8,9
# 服务器(同机另开终端;产物是纯 JSON)
python scripts/synth/writer_llm.py --task scenarios --n 100 --out scenarios.json
python scripts/synth/writer_llm.py --task wordings  --n 60  --out wordings.json
# 拷回本地 data/synth/ 后:
python scripts/synth/gen_v0.py --n 500 --group rthk --scenarios data/synth/scenarios.json --wordings data/synth/wordings.json
```
标签安全性:LLM 只供语言多样性;类别由代码注入、措辞过不变量校验器、
语言标签由字符组成推断(不信任模型自报)、加载时二次校验。

## 下一步(按 PLAN §7 M1)

1. 试录 2 对 × 3 session,走通"录→对齐→标→质检→入 harness",修订采集协议至 v1.0;
2. 探针音频渲染脚本(TTS 轨 + 真人录音轨;评测渲染与训练合成生成器隔离);
3. 快通路 v0(开源流式 VAD + 封闭集应声词表)接入 harness 出第一组基线数;
4. 韵律探针集(係啊/係咩/唔係啊 最小对 200–300 条)。
