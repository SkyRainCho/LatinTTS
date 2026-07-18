# 现代罗马教会式拉丁语发音规范基线

状态：阶段 1 规则基线。本文记录可追溯的发音政策；词级规范化和音节划分已实现，G2P 尚未实现。

## 目标与非目标

目标是为现代罗马教会式拉丁语建立确定、可审计的文本前端规范。每条规则应能追溯到来源登记中的稳定 `source_id` 和具体位置，并最终产生音节、重音、IPA 与规范音素。

本阶段不覆盖古典拉丁语、地区性教会读音、歌唱时值或声学模型训练，也不从候选录音反推规范。下述 Unicode 与拼写规范化契约及音节划分已有实现；G2P 尚未实现，未决行为必须显式保留，不能由实现自行猜测。

## 来源优先级

发生冲突时按来源登记中的 `authority_rank` 处理：

1. `liber-usualis-1962` 是罗马式礼仪发音的主要规则来源。
2. `ewtn-ecclesiastical-latin` 仅用于交叉核对解释措辞；`allen-greenough-accents` 提供重音规则；`perseus-lewis-short` 提供词汇级元音数量证据。
3. `wikimedia-ecclesiastical-pronunciation` 仅是评测音频索引，逐文件核对许可后才能使用。
4. `librivox-public-domain` 仅是候选语料索引，必须人工筛选读音并核对适用法域。

低优先级来源不能覆盖高优先级规则。候选音频不能自动升级为规范来源或训练数据。任何词典证据都要记录具体词条定位。

## Unicode 与拼写规范化

文本前端必须区分四种责任，不能用一种字符串兼任：

1. **原始短语与词面**：`original_text` 逐码位保留调用方输入。每个词的 `surface` 和半开区间 `source_span` 永久指向原文；它们是展示、诊断和人工覆盖的依据。
2. **短语规范文本**：`normalized_text` 只做 Unicode NFC 和空白整理，保留标点与原始正字法选择；它不作为词级 G2P 输入。
3. **词级 canonical normalized token**：`NormalizedWord.normalized` 和后续 `PronunciationToken.normalized` 可做 casefold、重音提示提取，以及非破坏的连字展开。`æ -> ae` 必须记录稳定 transformation ID `expand-ae-ligature`；`œ -> oe` 必须记录 `expand-oe-ligature`。长度变化绝不改写原文 span，`surface`、`source_span` 和 transformation IDs 共同提供可追踪的位置映射。G2P 和音节规则只消费这一层。
4. **lookup key**：在 canonical normalized token 之上产生，仅用于词典和例外查询。`j -> i` 记录 `lookup-j-to-i`，`v -> u` 记录 `lookup-v-to-u`；检索命中不得覆盖 canonical normalized token，也不能改变展示文本或原文 span。

连字上的附加符号采用可复现的工程归一化政策：先统一分解并展开基本连字，原附加符号自然保留在展开序列的第二个字母上，因此 `ǣ -> aē`；`ǽ` 的 acute 作为索引 `1` 的显式重音提示提取后从 canonical token 移除。lookup key 只移除 acute，保留 macron、diaeresis 及其他非 acute 附加符号。

因此 `æ/ae`、`œ/oe` 在 canonical token 层闭合到规则拼写，而 `j/i`、`u/v` 只在变体感知检索中归并。必须保留原始拼写和位置映射；禁止对原始短语做无条件全局替换，也禁止把 lookup key 写回任一文本层。

## 元音

下表的 IPA 是面向确定性 G2P 的宽式工程表示；来源用英语近似词描述音质，并明确指出精确音值最好通过听辨学习，因此这里不声明额外的开闭元音对立。

| 规则 | 规范输出 | 说明 | 来源 |
| --- | --- | --- | --- |
| `a` | `/a/` | 保持开放、饱满的单一音质，尤其在 `m/n` 前 | `liber-usualis-1962`, PDF lines 1254-1262 |
| `e` | `/e/` | 单一音质，不产生英语 *ray* 式滑音 | `liber-usualis-1962`, PDF lines 1254-1264 |
| `i` | `/i/` | 单一音质 | `liber-usualis-1962`, PDF lines 1254-1265 |
| `o` | `/o/` | 单一音质，不产生英语 *go* 式滑音 | `liber-usualis-1962`, PDF lines 1254-1266 |
| `u` | `/u/` | 单一音质 | `liber-usualis-1962`, PDF lines 1254-1267 |
| 元音长短 | 不改变规范音素 | 长短不能取代词重音；时值留给朗读或歌唱层 | `liber-usualis-1962`, PDF lines 1269-1272 |

## 双元音与相邻元音

G2P/音节规则消费展开后的 canonical normalized token：到达本节前，`æ` 已展开为 `ae`，`œ` 已展开为 `oe`，并分别保留 `expand-ae-ligature` 或 `expand-oe-ligature` transformation ID。因此连字拼写与双字母拼写走同一条 `ae/oe` 规则，但原始 `surface` 和 `source_span` 不变。

双元音判断逐码位比较 Unicode NFD base letter，而不是只比较字面拼写；因此 `ǣ` 展开所得 canonical `aē` 的 base-letter 序列仍是 `ae`，属于一个双元音核。任一相关元音带 diaeresis 时禁止合并，例如 `poëta -> ("po", "ë", "ta")`。音节边界始终是 canonical token 的 Python code-point 半开区间；`surface` 的原文跨度不在此阶段重算。

| 环境 | 音节与输出政策 | 来源 |
| --- | --- | --- |
| 一般相邻元音，包括 `ou`、`ai` | 各保留自己的音质并分属不同音节 | `liber-usualis-1962`, PDF lines 1273-1278 |
| `ae`、`oe` | 合为一个音节，规范输出 `/e/` | `liber-usualis-1962`, PDF lines 1279-1280 |
| `au`、`eu`、`ay` | 同属一个音节但两个元音都发出，重心在第一个元音 | `liber-usualis-1962`, PDF lines 1281-1289 |
| `ei` | 仅感叹词 `hei` 按一个音节处理；其他位置如 `mei` 分开 | `liber-usualis-1962`, PDF lines 1290-1291 |
| `qu` 或 `ngu` 后接元音 | `u` 保持音质，并与后续元音同属一个音节 | `liber-usualis-1962`, PDF lines 1292-1294 |
| `cui` | 通常为两个音节；诗歌格律要求的一音节用法是显式例外 | `liber-usualis-1962`, PDF lines 1294-1297 |

## 辅音

“前元音”在本表指 `e/ae/oe/i/y`。

| 模式 | 规范输出 | 条件或说明 | 来源 |
| --- | --- | --- | --- |
| `c` | `/tʃ/` 或 `/k/` | 前元音前为 `/tʃ/`，其他位置为 `/k/` | `liber-usualis-1962`, PDF lines 1298-1302, 1307-1308 |
| `cc` | `/t.tʃ/` | 前元音前清楚保留闭塞后接 `/tʃ/` | `liber-usualis-1962`, PDF lines 1303-1304 |
| `sc` | `/ʃ/` | 前元音前 | `liber-usualis-1962`, PDF lines 1305-1306 |
| `ch` | `/k/` | 包括 `e/i` 前 | `liber-usualis-1962`, PDF lines 1309-1310 |
| `g` | `/dʒ/` 或 `/g/` | 前元音前为 `/dʒ/`，其他位置为 `/g/` | `liber-usualis-1962`, PDF lines 1311-1314 |
| `gn` | `/ɲ/` | 采用来源所述意大利语式软音的宽式表示 | `liber-usualis-1962`, PDF lines 1315-1318 |
| `h` | `/k/` 或不输出 | `nihil`、`mihi` 及其派生词中为 `/k/`，其他位置不输出 | `liber-usualis-1962`, PDF lines 1319-1321 |
| 辅音 `j`（也可写作 `i`） | `/j/` | 与后续元音形成一个连贯发音 | `liber-usualis-1962`, PDF lines 1322-1324 |
| `r` | `/r/` | 与其他辅音相邻时也不能省略；朗读实现应轻触或轻颤 | `liber-usualis-1962`, PDF lines 1325-1331 |
| `s` | `/s/` | 元音间仍保持完整音位 `/s/`；“轻微软化”只作朗读实现注释 | `liber-usualis-1962`, PDF lines 1332-1334 |
| `ti` + 元音 | `/tsi/` | 前一字母不是 `s/x/t` 时；否则 `t` 保持 `/t/` | `liber-usualis-1962`, PDF lines 1335-1341 |
| `th` | `/t/` | 不保留送气对立 | `liber-usualis-1962`, PDF line 1342 |
| `x` | `/ks/` | 元音间的“轻微软化”不改变阶段 1 完整音位输出 | `liber-usualis-1962`, PDF lines 1343-1344 |
| `xc` | `/kʃ/` 或 `/ksk/` | 前元音前为 `/kʃ/`，其他元音前保留硬音组合 | `liber-usualis-1962`, PDF lines 1345-1348 |
| `y` | `/i/` | 作为元音处理 | `liber-usualis-1962`, PDF line 1349 |
| `z` | `/dz/` | 采用来源给出的塞擦音描述 | `liber-usualis-1962`, PDF line 1350 |
| `b/d/f/k/l/m/n/p/q/v` | 对应基础辅音 | 阶段 1 保留独立、清楚的辅音发音 | `liber-usualis-1962`, PDF lines 1351-1354 |

元音间 `s` 的完整音位输出在本基线中必须保持 `/s/`。来源的“轻微软化”不等同于已经证明的 `/z/`；只有获得更精确且不冲突的来源后，才能通过冲突决策记录改变该音位。

## 音节划分

音节器只接收 `NormalizedWord.normalized`，不重复规范化，也不读取词典。它先识别音节核，再在相邻音节核之间分配辅音；`syllable_ranges()` 返回 canonical Python code-point 半开区间，`syllabify()` 只按这些区间切片。

| 规则或固定集合 | 处理与例子 | 来源 |
| --- | --- | --- |
| 每个音节都完整发音 | 不得吞掉或截短弱 penult，例如不得把 `Domine` 读成 `Domne` | `liber-usualis-1962`, PDF lines 1231-1247 |
| 元音核固定集合 | `a e i o u y ā ē ī ō ū ȳ`；附加符号不改变 base-letter 元音身份。`a/e/i/o/u` 与长短政策见元音段，`y` 作为元音 | `liber-usualis-1962`, PDF lines 1254-1272, 1349 |
| 双元音固定集合 | `ae oe au eu ay`；按 Unicode base letters 比较，diaeresis 打断合并。`caelum -> ("cae", "lum")`、canonical `aēlum -> ("aē", "lum")`、`poëta -> ("po", "ë", "ta")` | `liber-usualis-1962`, PDF lines 1273-1289；Unicode/diaeresis 是 canonical token 工程契约 |
| 辅音 `i` | 词首接元音或位于两个元音之间时作为下一音节的辅音起始；带 diaeresis 时仍为元音。`alleluia -> ("al", "le", "lu", "ia")` | `liber-usualis-1962`, PDF lines 1322-1324 |
| `u` 滑音 | `q` 或 `ng` 后且后接元音时不另立音节核；带 diaeresis 时仍为元音，`qüi -> ("qü", "i")`。`qui -> ("qui",)`；来源明确规定的 `cui -> ("cu", "i")` 保持两音节 | `liber-usualis-1962`, PDF lines 1292-1297；diaeresis 是 canonical token 工程契约 |
| 允许的 onset 固定集合 | `bl br cl cr dr fl fr gl gr pl pr tr qu gu ch ph th gn` 整体进入下一音节，例如 `patris -> ("pa", "tris")`。这是阶段 1 的确定性工程 whitelist；其特殊字母组的发音身份分别按辅音表保留 | `liber-usualis-1962`, PDF lines 1292-1294, 1309-1318, 1342, 1351-1354；`ewtn-ecclesiastical-latin`, pronunciation tables |
| 单辅音 | 相邻音节核间的单辅音进入下一音节：`ave -> ("a", "ve")`、`gratia -> ("gra", "ti", "a")` | 阶段 1 确定性工程边界政策；元音与辅音身份来自 `liber-usualis-1962`, PDF lines 1254-1354 |
| 双辅音 | 从中间分开并保留两个辅音位置：`ecce -> ("ec", "ce")` | `liber-usualis-1962`, PDF lines 1352-1354 |
| 其他多辅音簇 | 仅把允许的最长 onset 后缀移到下一音节，否则只移最后一个辅音：`sanctus -> ("sanc", "tus")` | 阶段 1 确定性工程边界政策；辅音完整发音依据 `liber-usualis-1962`, PDF lines 1351-1354 |
| 相邻独立元音核 | 核之间没有辅音时直接在两核之间分界：`gratia -> ("gra", "ti", "a")` | `liber-usualis-1962`, PDF lines 1273-1278 |

`syllable_ranges("poëta") == ((0, 2), (2, 3), (3, 5))` 展示 canonical code-point 半开区间契约。无音节核的输入会失败并报告 `word contains no vowel nucleus`；调用方必须先完成词级 canonical 规范化。

## 重音

发出每个音节的完整音值不等于给每个音节相同重音；弱音节不得被删除，音长也不得自动视为重音。

| 词形条件 | 重音位置 | 来源 |
| --- | --- | --- |
| 每个音节 | 完整发音，同时保持重音与音长的区别 | `liber-usualis-1962`, PDF lines 1231-1247 |
| 双音节词 | 第一音节 | `allen-greenough-accents`, Section 12 |
| 三音节及以上，penult 为重音节 | penult | `allen-greenough-accents`, Section 12 |
| 三音节及以上，penult 为轻音节 | antepenult | `allen-greenough-accents`, Section 12 |
| 附着词加入词干 | 附着词之前的音节，不论该音节长短 | `allen-greenough-accents`, Section 12 |

penult 的轻重需要词汇数量或音节结构证据。`perseus-lewis-short` 可提供词汇证据，但每个词条必须记录具体定位。三音节及以上词若普通拼写无法证明 penult 轻重，只能生成候选，且必须返回 `PRONUNCIATION_NEEDS_REVIEW` 诊断；不得静默宣称确定，也不得把候选重音升级为规范输出。双音节词和有充分证据的词仍按上表确定重音。

## 双辅音

| 规则 | 规范表示 | 来源 |
| --- | --- | --- |
| 拼写双辅音 | 保留为两个辅音位置或等价的长辅音，不得简化为单辅音 | `liber-usualis-1962`, PDF lines 1352-1354 |
| `cc` + 前元音 | 使用辅音表中的 `/t.tʃ/`，不能先按普通双辅音简化 | `liber-usualis-1962`, PDF lines 1303-1304 |

声学层可选择闭塞延长或跨音节重接等实现，但必须保持与单辅音的可辨对立。

## 词间处理

阶段 1 不引入跨词同化、连诵改写或自动省音。词边界应保留在中间表示中，短语韵律不能改变词内规范音素。朗读或歌唱需要换气时，不得在一个词的新音节之前切断该词；此原则来自 `liber-usualis-1962`, PDF lines 1355-1357。

## 例外

例外必须以词条或受限上下文显式建模，并记录来源：

- `nihil`、`mihi` 及其派生词中的 `h` 输出 `/k/`；其他 `h` 不输出。来源：`liber-usualis-1962`, PDF lines 1319-1321。
- 感叹词 `hei` 的 `ei` 同属一个音节，其他词中的同形序列默认分开。来源：`liber-usualis-1962`, PDF lines 1290-1291。
- `cui` 通常为两个音节；只有记录到具体赞美诗格律要求时才允许一音节覆盖。来源：`liber-usualis-1962`, PDF lines 1294-1297。
- `ti` 规则在前一字母为 `s/x/t` 时不触发。来源：`liber-usualis-1962`, PDF lines 1335-1341。
- Allen and Greenough Section 12 所列重音例外尚未进入首批词典；录入时必须逐条保留定位。来源：`allen-greenough-accents`, Section 12。

## IPA 与规范音素表

阶段 1 采用宽式 IPA 作为人类可读表示，并以同一符号集作为规范音素表的起点。这是**有来源的工程归一化**：表中来源给出音质、发音方式或重音规则，规范音素则把这些描述映射到稳定的宽式内部类别，并非声称来源逐字使用了相同 IPA。`/e/`、`/o/` 是有来源支持的宽式类别，不声称排除实际朗读中的开闭变体。

| 类别 | 规范音素 | 来源 |
| --- | --- | --- |
| 元音 | `a e i o u` | `liber-usualis-1962`, PDF lines 1254-1272 |
| 滑音 | `j` | `liber-usualis-1962`, PDF lines 1322-1324 |
| 塞音 | `p b t d k g` | `liber-usualis-1962`, PDF lines 1298-1314, 1341-1342, 1351 |
| 塞擦音 | `ts dz tʃ dʒ` | `liber-usualis-1962`, PDF lines 1301-1304, 1311-1312, 1335-1340, 1350 |
| 擦音 | `f v s ʃ` | `liber-usualis-1962`, PDF lines 1305-1306, 1332-1334, 1351 |
| 鼻音 | `m n ɲ` | `liber-usualis-1962`, PDF lines 1315-1318, 1351 |
| 流音 | `l r` | `liber-usualis-1962`, PDF lines 1325-1331, 1351 |
| 组合 | `ks kʃ ksk`，以及按顺序保留的双辅音 | `liber-usualis-1962`, PDF lines 1343-1354 |
| 韵律标记 | IPA 主重音 `ˈ`；音节边界 `.` 仅用于表示，不作为声学音素 | `liber-usualis-1962`, PDF lines 1231-1247；`allen-greenough-accents`, Section 12 |

规范音素输出必须与规则追踪信息分离：规则来源、例外来源和“轻微软化”等朗读注释不能伪装成音素。

## 规则覆盖矩阵

| 范围 | 规范来源 | 阶段 1 状态 | 后续验证 |
| --- | --- | --- | --- |
| 单元音 | `liber-usualis-1962`, PDF lines 1254-1272 | 基线已记录 | 用人工筛选录音核对音质范围 |
| 双元音与相邻元音 | `liber-usualis-1962`, PDF lines 1273-1297 | 基线已记录 | 为每个合并、分离与例外写 G2P 测试 |
| `c/cc/sc/ch/g/gn/h/j` | `liber-usualis-1962`, PDF lines 1298-1324 | 基线已记录 | 增加上下文边界与大小写测试 |
| `r/s/ti/th/x/xc/y/z` | `liber-usualis-1962`, PDF lines 1325-1350 | 基线已记录 | 验证例外优先级与宽式 IPA |
| 其他辅音与双辅音 | `liber-usualis-1962`, PDF lines 1351-1354 | 基线已记录 | 定义音节重接和模型音素编码 |
| 完整音节与重音区别 | `liber-usualis-1962`, PDF lines 1231-1247 | 政策已记录 | 建立弱 penult 不丢失的回归样例 |
| 重音位置 | `allen-greenough-accents`, Section 12 | 基线及无证据诊断已记录 | 接入词汇数量并覆盖例外 |
| Unicode 与拼写规范化 | 工程文本契约 | phrase NFC、词级连字展开、lookup 变体和 span 追踪已实现 | 在 G2P 集成中保持四层文本契约 |
| 词间音变 | 尚无规范规则 | 明确禁用 | 仅在新增不冲突来源后提案 |

## 冲突决策记录

| 日期 | 议题 | 决策 | 依据 |
| --- | --- | --- | --- |
| 2026-07-17 | 元音间 `s` 的“轻微软化”是否等同 `/z/` | 否。完整音位保持 `/s/`，软化仅为朗读实现注释 | `liber-usualis-1962`, PDF lines 1332-1334 未给出 `/z/` 等值；避免无证据扩大解释 |
| 2026-07-18 | Perseus Lewis and Short 条款版本 | 登记为 `CC BY-SA 4.0`，保留 Perseus 署名、可用性声明及修改回馈要求 | `perseus-lewis-short` 指定目录的当前 `README.md` 许可段；取代过时的 3.0 元数据 |
| 2026-07-18 | 拼写变体在哪一层归并 | 原始 `surface/source_span` 不变；词级 canonical token 展开 `æ/œ` 并记录 transformation ID；lookup key 再归并 `j/i`、`v/u` | 使 `ae/oe` G2P 规则可消费连字，同时避免全局替换破坏拼写与 span |
| 2026-07-18 | penult 轻重缺少证据时是否给出确定重音 | 否。仅生成候选并返回 `PRONUNCIATION_NEEDS_REVIEW` | `allen-greenough-accents`, Section 12 需要 penult 轻重；普通拼写不总能提供该证据 |

后续冲突记录必须包含日期、候选解释、采用结果和精确来源位置。改变既有规范音素属于可审计的规则版本变更。
