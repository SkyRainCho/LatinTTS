# Gold pronunciation fixture

`gold_pronunciations.jsonl` 是现代罗马教会式拉丁语的人工批准回归集。它用于检测实现是否偏离已审核标注，不独立证明发音正确；规范证据和逐词审核定位见 `docs/pronunciation/roman-ecclesiastical.md`。

关键字段语义：

- `ipa`: human-verified, not pipeline-derived。管线可以辅助生成候选，但最终值必须由人工对照登记来源和精确 locator 后批准，不能用 `Pronouncer` 当前输出自动生成并把同一输出当成验证证据。
- `syllables` 与 `stress_index`：与 `ipa` 一起人工核验的规范标注，不是从当前实现重新导出的缓存。
- `rule_ids` 与 `source_ids`：记录批准时适用的完整规则和来源集合，用于定位规则变更的影响范围。
- `review_state="approved"`：表示该行已经完成来源核对；不是 `audit.py` 自动推断出的正确性结论。

规则或例外发生变化时，先识别所有受影响行并重新对照来源，人工批准后才可修改本文件。不得用批量重生成掩盖实现与已审核标注之间的差异。
