# 现代罗马教会式拉丁语发音规范基线

状态：阶段 1 规则基线。本文记录可追溯的发音政策；词级规范化、音节划分、来源可追溯的重音解析和罗马教会式 G2P 已实现。

## 目标与非目标

目标是为现代罗马教会式拉丁语建立确定、可审计的文本前端规范。每条规则应能追溯到来源登记中的稳定 `source_id` 和具体位置，并最终产生音节、重音、IPA 与规范音素。

本阶段不覆盖古典拉丁语、地区性教会读音、歌唱时值或声学模型训练，也不从候选录音反推规范。下述 Unicode 与拼写规范化契约、音节划分和来源索引 G2P 已有实现；未决行为必须显式保留，不能由实现自行猜测。

## 来源优先级

发生冲突时按来源登记中的 `authority_rank` 处理：

1. `liber-usualis-1962` 是罗马式礼仪发音的主要规则来源。
2. `ewtn-ecclesiastical-latin` 仅用于交叉核对解释措辞；`iveson-roman-pronunciation-1964` 为 `ph -> /f/` 提供直接次级证据；`allen-greenough-accents` 提供重音规则；`perseus-lewis-short` 提供词汇级元音数量证据。
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
| `ph` | `/f/` | 来源明确写作 “PH — as the letter F”；不把 Liber PDF line 1351 的单字母 `P` 冒充直接证据 | `iveson-roman-pronunciation-1964`, PDF page 1 (printed p. 14), lines 44-46 |
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

G2P 扫描器按上表从特殊二合字到单字符执行当前位置最长匹配，并为每个输出音素保留 canonical word source index。它不做整词连续替换；因此 `ecce` 的 `cc` 分别把 `/t/` 归到 `ec`、把 `/t͡ʃ/` 归到 `ce`，输出 `ˈet.t͡ʃe`。`excelsis` 的 `xc` 同理跨 `ex/cel` 归属。macron 只参与 base-letter 折叠而不改变 source index；diaeresis 会阻止 `ae/oe/au/eu/ay`、辅音 `i`、`qu` 和 `ngu` 的合并规则。

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

实现严格按以下优先级解析：请求覆盖 > acute 显式提示 > 全词例外/词典 > 经已知 base lexicon 证实的附着词 > 单/双音节规则 > 可证明的 heavy penult > antepenult 候选。附着词只在去掉 `que/ne/ve` 后的 base 已存在于词典时触发；仅仅以这些字母结尾不构成证据。可证明的 heavy penult 仅包括含 macron、含 base-letter 双元音，或以辅音结尾的音节。不得把未知开放 penult 静默确定；此时只给出 antepenult 候选并附带 `PRONUNCIATION_NEEDS_REVIEW`。

首批重音词典的 locator 与推导如下。Lewis and Short 的 `entryFree id` 和 `key` 均指 `perseus-lewis-short` 登记的 XML；Liber 页码同时给出 PDF 页和印刷页。`benedictus`、`excelsis` 是明确标注的 lemma/inflection + Section 12 闭 penult 规则推导，不声称词典直接印出了屈折词重音。

| lookup key | 重音 | 证据与具体定位 |
| --- | --- | --- |
| `dominus` | `do` | `perseus-lewis-short`, `entryFree id=n14699`, `key=dominus`, `dŏmĭnus`；`allen-greenough-accents`, Section 12 |
| `regina` | `gi` | `perseus-lewis-short`, `entryFree id=n40899`, `key=regina`, `rēgīna`；`allen-greenough-accents`, Section 12 |
| `maria` | `ri` | `perseus-lewis-short`, `entryFree id=n28037`, `key=Maria1`, `Mărī^a`, sense I.1 Mary；`allen-greenough-accents`, Section 12 |
| `gratia` | `gra` | `perseus-lewis-short`, `entryFree id=n19896`, `key=gratia`, `grātĭa`；`allen-greenough-accents`, Section 12 |
| `gloria` | `glo` | `perseus-lewis-short`, `entryFree id=n19675`, `key=gloria`, `glōrĭa`；`allen-greenough-accents`, Section 12，轻 penult -> antepenult |
| `kyrie` | `ky` | `liber-usualis-1962`, PDF page 32 (printed xxxviii), R example，印作 `Kýrie` |
| `caelum` | `cae` | `perseus-lewis-short`, `entryFree id=n6042`, `key=caelum2`；`allen-greenough-accents`, Section 12 双音节规则 |
| `alleluia` | `lu` | `perseus-lewis-short`, `entryFree id=n1926`, `key=alleluja`, `allēlūja`；`liber-usualis-1962`, PDF page 32 (printed xxxviii), J example；`allen-greenough-accents`, Section 12 |
| `magnificat` | `gni` | `perseus-lewis-short`, `entryFree id=n27636`, `key=magnifico`, `magnĭfĭco` 的现在时第三人称单数；`liber-usualis-1962`, PDF page 32 (printed xxxviii), GN example；`allen-greenough-accents`, Section 12 |
| `misericordia` | `cor` | `perseus-lewis-short`, `entryFree id=n29266`, `key=misericordia`, `mĭsĕrĭcordĭa`；`liber-usualis-1962`, PDF page 32 (printed xxxviii), S example；`allen-greenough-accents`, Section 12 |
| `benedictus` | `dic` | `perseus-lewis-short`, `entryFree id=n5170`, `key=benedico`, principal part `ctum` 支持该分词；`allen-greenough-accents`, Section 12，`dic` 为闭 penult 的规则推导 |
| `excelsis` | `cel` | `perseus-lewis-short`, `entryFree id=n16651`, `key=excelsus`, inflection `a, um` 支持该屈折词；`allen-greenough-accents`, Section 12，`cel` 为闭 penult 的规则推导 |

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

- 打包例外表当前只收录 `nihil`、`mihi`，其中 `h` 输出 `/k/`；其他 `h` 不输出。派生词只有在逐条确认来源并加入例外表后才能获得 `/k/`，扫描器不得仅凭词形猜测。来源：`liber-usualis-1962`, PDF lines 1319-1321。
- 感叹词 `hei` 的 `ei` 同属一个音节，其他词中的同形序列默认分开。来源：`liber-usualis-1962`, PDF lines 1290-1291。
- `cui` 通常为两个音节；只有记录到具体赞美诗格律要求时才允许一音节覆盖。来源：`liber-usualis-1962`, PDF lines 1294-1297。
- `ti` 规则在前一字母为 `s/x/t` 时不触发。来源：`liber-usualis-1962`, PDF lines 1335-1341。
- Allen and Greenough Section 12 所列重音例外尚未进入首批词典；录入时必须逐条保留定位。来源：`allen-greenough-accents`, Section 12。

## IPA 与规范音素表

阶段 1 采用宽式 IPA 作为人类可读表示，并以拆分后的同一符号集作为规范音素表的起点。这是**有来源的工程归一化**：表中来源给出音质、发音方式或重音规则，规范音素则把这些描述映射到稳定的宽式内部类别，并非声称来源逐字使用了相同 IPA。`/e/`、`/o/` 是有来源支持的宽式类别，不声称排除实际朗读中的开闭变体。

| 类别 | 规范音素 | 来源 |
| --- | --- | --- |
| 元音 | `a e i o u` | `liber-usualis-1962`, PDF lines 1254-1272 |
| 滑音 | `i̯ u̯ j w` | `liber-usualis-1962`, PDF lines 1281-1294, 1322-1324 |
| 塞音 | `p b t d k ɡ` | `liber-usualis-1962`, PDF lines 1298-1314, 1341-1342, 1351 |
| 塞擦音 | `t͡s d͡z t͡ʃ d͡ʒ` | `liber-usualis-1962`, PDF lines 1301-1304, 1311-1312, 1335-1340, 1350 |
| 擦音 | `f v s ʃ` | `liber-usualis-1962`, PDF lines 1305-1306, 1332-1334, 1351 |
| 鼻音 | `m n ŋ ɲ` | `liber-usualis-1962`, PDF lines 1292-1294, 1315-1318, 1351 |
| 流音 | `l r` | `liber-usualis-1962`, PDF lines 1325-1331, 1351 |
| 组合 | `x -> (k, s)`、`xc -> (k, ʃ)`，以及按顺序保留的双辅音；模型 token 不把组合粘成一个音素 | `liber-usualis-1962`, PDF lines 1343-1354 |
| 韵律标记 | IPA 主重音 `ˈ` 在非首重读音节前替代该处的 `.`，如 `deˈʃen.dit`；模型序列仍显式输出独立 token `"."`, `"ˈ"` | `liber-usualis-1962`, PDF lines 1231-1247；`allen-greenough-accents`, Section 12 |

规范音素输出必须与规则追踪信息分离：规则来源、例外来源和“轻微软化”等朗读注释不能伪装成音素。`.` 与 `ˈ` 是模型输入格式的边界/韵律 token，不属于 `PHONEME_INVENTORY`；阶段 1 不实现 Piper 专用映射。

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

### 元音与双元音黄金批次审核矩阵

以下四批每批 10 条。审核时逐词运行 `Pronouncer`，只批准非 `ResolutionMethod.CANDIDATE` 且 `warnings == ()` 的结果；黄金记录中的 `rule_ids` 与 `source_ids` 按 token 实际集合完整登记。表中列出本批关注的关键规则，词内其他基础辅音规则仍按下节的实现规则表核对。所有单音节、双音节重音均直接依据 `allen-greenough-accents`, Section 12；`Raymundus` 的闭 penult `mun` 也按同一节推导，不需要扩大重音词典。

#### A：短/长书写元音与 `y`

| word | category | 关键 rule IDs | 精确来源 locator | review |
| --- | --- | --- | --- | --- |
| `māter` | `vowels` | `disyllable-stress`, `simple-a` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `mē` | `vowels` | `monosyllable-stress`, `simple-e` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `dōnum` | `vowels` | `disyllable-stress`, `simple-o`, `simple-u` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `fīdes` | `vowels` | `disyllable-stress`, `simple-i`, `simple-e` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `lūmen` | `vowels` | `disyllable-stress`, `simple-u`, `simple-e` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `rosa` | `vowels` | `disyllable-stress`, `simple-o`, `simple-a` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `tibi` | `vowels` | `disyllable-stress`, `simple-i` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `bonus` | `vowels` | `disyllable-stress`, `simple-o`, `simple-u` | `liber-usualis-1962`, PDF lines 1254-1272；`allen-greenough-accents`, Section 12 | approved |
| `hymnus` | `vowels` | `disyllable-stress`, `y-as-i`, `h-muted` | `liber-usualis-1962`, PDF lines 1319-1321, 1349；`allen-greenough-accents`, Section 12 | approved |
| `myrrha` | `vowels` | `disyllable-stress`, `y-as-i`, `h-muted` | `liber-usualis-1962`, PDF lines 1319-1321, 1325-1331, 1349；`allen-greenough-accents`, Section 12 | approved |

#### B：相邻元音分别成音节

| word | category | 关键 rule IDs | 精确来源 locator | review |
| --- | --- | --- | --- | --- |
| `mea` | `vowels` | `disyllable-stress`, `simple-e`, `simple-a` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `tua` | `vowels` | `disyllable-stress`, `simple-u`, `simple-a` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `sua` | `vowels` | `disyllable-stress`, `simple-u`, `simple-a` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `duo` | `vowels` | `disyllable-stress`, `simple-u`, `simple-o` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `via` | `vowels` | `disyllable-stress`, `simple-i`, `simple-a` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `pia` | `vowels` | `disyllable-stress`, `simple-i`, `simple-a` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `pius` | `vowels` | `disyllable-stress`, `simple-i`, `simple-u` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `mei` | `vowels` | `disyllable-stress`, `simple-e`, `simple-i` | `liber-usualis-1962`, PDF lines 1290-1291；`allen-greenough-accents`, Section 12 | approved |
| `prout` | `vowels` | `disyllable-stress`, `simple-o`, `simple-u` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |
| `ait` | `vowels` | `disyllable-stress`, `simple-a`, `simple-i` | `liber-usualis-1962`, PDF lines 1273-1278；`allen-greenough-accents`, Section 12 | approved |

#### C：`ae` 与 `oe`

| word | category | 关键 rule IDs | 精确来源 locator | review |
| --- | --- | --- | --- | --- |
| `laetus` | `diphthongs` | `disyllable-stress`, `ae-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `maestus` | `diphthongs` | `disyllable-stress`, `ae-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `praeda` | `diphthongs` | `disyllable-stress`, `ae-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `caecus` | `diphthongs` | `disyllable-stress`, `c-before-front-vowel`, `ae-e` | `liber-usualis-1962`, PDF lines 1279-1280, 1301-1302；`allen-greenough-accents`, Section 12 | approved |
| `aedes` | `diphthongs` | `disyllable-stress`, `ae-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `poena` | `diphthongs` | `disyllable-stress`, `oe-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `foedus` | `diphthongs` | `disyllable-stress`, `oe-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `coena` | `diphthongs` | `disyllable-stress`, `c-before-front-vowel`, `oe-e` | `liber-usualis-1962`, PDF lines 1279-1280, 1301-1302；`allen-greenough-accents`, Section 12 | approved |
| `foenum` | `diphthongs` | `disyllable-stress`, `oe-e` | `liber-usualis-1962`, PDF lines 1279-1280；`allen-greenough-accents`, Section 12 | approved |
| `coepit` | `diphthongs` | `disyllable-stress`, `c-before-front-vowel`, `oe-e` | `liber-usualis-1962`, PDF lines 1279-1280, 1301-1302；`allen-greenough-accents`, Section 12 | approved |

#### D：`au/eu/ay` 与 `qu/ngu/cui` 边界

| word | category | 关键 rule IDs | 精确来源 locator | review |
| --- | --- | --- | --- | --- |
| `causa` | `diphthongs` | `disyllable-stress`, `au-diphthong` | `liber-usualis-1962`, PDF lines 1281-1289；`allen-greenough-accents`, Section 12 | approved |
| `aura` | `diphthongs` | `disyllable-stress`, `au-diphthong` | `liber-usualis-1962`, PDF lines 1281-1289；`allen-greenough-accents`, Section 12 | approved |
| `neuter` | `diphthongs` | `disyllable-stress`, `eu-diphthong` | `liber-usualis-1962`, PDF lines 1281-1289；`allen-greenough-accents`, Section 12 | approved |
| `heu` | `diphthongs` | `monosyllable-stress`, `eu-diphthong` | `liber-usualis-1962`, PDF lines 1281-1289；`allen-greenough-accents`, Section 12 | approved |
| `Raymundus` | `diphthongs` | `heavy-penult-stress`, `ay-diphthong` | `liber-usualis-1962`, PDF lines 1281-1289；`allen-greenough-accents`, Section 12，闭 penult `mun` | approved |
| `quo` | `diphthongs` | `monosyllable-stress`, `qu-before-vowel` | `liber-usualis-1962`, PDF lines 1292-1294；`allen-greenough-accents`, Section 12 | approved |
| `aqua` | `diphthongs` | `disyllable-stress`, `qu-before-vowel` | `liber-usualis-1962`, PDF lines 1292-1294；`allen-greenough-accents`, Section 12 | approved |
| `sanguis` | `diphthongs` | `disyllable-stress`, `ngu-before-vowel` | `liber-usualis-1962`, PDF lines 1292-1294；`allen-greenough-accents`, Section 12 | approved |
| `lingua` | `diphthongs` | `disyllable-stress`, `ngu-before-vowel` | `liber-usualis-1962`, PDF lines 1292-1294；`allen-greenough-accents`, Section 12 | approved |
| `cuius` | `diphthongs` | `disyllable-stress`, `simple-u`, `i-consonantal` | `liber-usualis-1962`, PDF lines 1294-1297, 1322-1324；`allen-greenough-accents`, Section 12 | approved |

既有黄金记录 `qui`（`qu-before-vowel`）与 `cui`（`simple-u` + `simple-i`）保持不变，并与本批 `quo`、`aqua`、`sanguis`、`lingua`、`cuius` 共同锁定 `qu/ngu/cui` 的合并与非合并边界。带 diaeresis 的人工变体留待专门的拼写变体批次，不在本批升级为 source-backed gold。

### G2P 实现规则与 locator

下表列出阶段 1 当前实现的全部稳定 rule ID。正例只说明规则触发；反例用于锁定最长匹配或例外优先级。一个 locator 没有直接写出某工程 IPA token 时，表中只把来源描述映射到宽式音素，不声称来源使用了 IPA。

| rule IDs | 条件与输出 | 正例 / 反例 | 来源 locator |
| --- | --- | --- | --- |
| `simple-a`, `simple-e`, `simple-i`, `simple-o`, `simple-u`, `y-as-i` | 单元音输出 `a/e/i/o/u`；`y -> i` | `pater`, `poëta`, `kyrie` | `liber-usualis-1962`, PDF lines 1254-1272, 1349 |
| `ae-e`, `oe-e` | 无 diaeresis 的 base-letter `ae/oe -> e` | `caelum`, `poena` / `poëta` | `liber-usualis-1962`, PDF lines 1279-1280 |
| `au-diphthong`, `eu-diphthong`, `ay-diphthong` | 无 diaeresis 时输出 `a,u̯`、`e,u̯`、`a,i̯` | `lauda`, `euge`, `Raymundus` / `aüla` | `liber-usualis-1962`, PDF lines 1281-1289 |
| `qu-before-vowel`, `ngu-before-vowel` | 无 diaeresis 时 `qu -> k,w`；`ngu -> ŋ,ɡ,w` | `qui`, `sanguis` / `qüi`, `sangüis` | `liber-usualis-1962`, PDF lines 1292-1294 |
| `c-before-front-vowel`, `c-hard` | `c` 在 `e/ae/oe/i/y` 前输出 `t͡ʃ`，否则 `k` | `caelum` / `caritas` | `liber-usualis-1962`, PDF lines 1301-1302, 1307-1308 |
| `cc-before-front-vowel` | 同一前元音环境输出 `t,t͡ʃ`，并按两个 `c` 的 source index 分属音节 | `ecce -> ˈet.t͡ʃe` / `siccus` | `liber-usualis-1962`, PDF lines 1303-1304 |
| `sc-before-front-vowel` | 同一前元音环境输出 `ʃ` | `descendit` / `scutum` | `liber-usualis-1962`, PDF lines 1305-1306 |
| `ch-hard` | `ch -> k`，包括 `e/i` 前 | `Cham`, `machina` / 不走 `c-before-front-vowel` | `liber-usualis-1962`, PDF lines 1309-1310 |
| `g-before-front-vowel`, `g-hard` | `g` 在前元音前输出 `d͡ʒ`，否则 `ɡ` | `regina` / `ego` | `liber-usualis-1962`, PDF lines 1311-1314 |
| `gn-palatal` | `gn -> ɲ` | `regnum` / 不拆成 `ɡ,n` | `liber-usualis-1962`, PDF lines 1315-1318 |
| `h-mihi-nihil`, `h-muted` | 仅打包例外或后续逐条确认的例外输出 `k`；普通 `h` 不输出 | `mihi`, `nihil` / `hora` | `liber-usualis-1962`, PDF lines 1319-1321 |
| `i-consonantal`, `j-consonantal` | 词首接元音或元音间的 `i`，以及显式 `j` 输出 `j`；diaeresis 阻止辅音 `i` | `iam`, `major` / `aïa` | `liber-usualis-1962`, PDF lines 1322-1324 |
| `simple-r` | `r -> r`，不能在辅音旁省略 | `carnis` | `liber-usualis-1962`, PDF lines 1325-1331 |
| `simple-s` | 阶段 1 始终输出完整音位 `s` | `misericordia` / 不自动改写为 `z` | `liber-usualis-1962`, PDF lines 1332-1334 |
| `ti-before-vowel`, `simple-t` | `ti` 后接元音且前一字母不是 `s/x/t` 时输出 `t͡s,i`；否则 `t,i` | `gratia` / `hostia`, `mixtio`, `attia` | `liber-usualis-1962`, PDF lines 1335-1341 |
| `th-t` | `th -> t` | `Thomas` | `liber-usualis-1962`, PDF line 1342 |
| `x-ks`, `xc-before-front-vowel` | `x -> k,s`；前元音前 `xc -> k,ʃ`，两个输出按 source index 分属音节 | `exercitus`, `excelsis` / `excussorum` | `liber-usualis-1962`, PDF lines 1343-1348 |
| `z-dz` | `z -> d͡z` | `zizania` | `liber-usualis-1962`, PDF line 1350 |
| `simple-b`, `simple-d`, `simple-f`, `simple-k`, `simple-l`, `simple-m`, `simple-n`, `simple-p`, `q-hard`, `simple-v` | 其余基础辅音逐字输出；无后续元音的 `q` 退化为 `k` | 基础拼写 / `qu` 优先走最长匹配 | `liber-usualis-1962`, PDF line 1351 |
| `ph-f` | `ph -> f` | `phonascus`, `phasma` / 不拆为 `p` 加静音 `h` | `iveson-roman-pronunciation-1964`, PDF page 1 (printed p. 14), lines 44-46 |

## 冲突决策记录

| 日期 | 议题 | 决策 | 依据 |
| --- | --- | --- | --- |
| 2026-07-17 | 元音间 `s` 的“轻微软化”是否等同 `/z/` | 否。完整音位保持 `/s/`，软化仅为朗读实现注释 | `liber-usualis-1962`, PDF lines 1332-1334 未给出 `/z/` 等值；避免无证据扩大解释 |
| 2026-07-18 | Perseus Lewis and Short 条款版本 | 登记为 `CC BY-SA 4.0`，保留 Perseus 署名、可用性声明及修改回馈要求 | `perseus-lewis-short` 指定目录的当前 `README.md` 许可段；取代过时的 3.0 元数据 |
| 2026-07-18 | 拼写变体在哪一层归并 | 原始 `surface/source_span` 不变；词级 canonical token 展开 `æ/œ` 并记录 transformation ID；lookup key 再归并 `j/i`、`v/u` | 使 `ae/oe` G2P 规则可消费连字，同时避免全局替换破坏拼写与 span |
| 2026-07-18 | penult 轻重缺少证据时是否给出确定重音 | 否。仅生成候选并返回 `PRONUNCIATION_NEEDS_REVIEW` | `allen-greenough-accents`, Section 12 需要 penult 轻重；普通拼写不总能提供该证据 |
| 2026-07-18 | `ph -> /f/` 的直接来源如何补齐 | 使用 Iveson 明确的 `PH — as the letter F`；`ph-f` 只引用该来源，词内其他基础规则仍分别引用 Liber，结果聚合实际命中的来源 | `iveson-roman-pronunciation-1964`, PDF page 1 (printed p. 14), lines 44-46；文首 lines 2-4 说明规则基于罗马省神职人员实际读音 |

后续冲突记录必须包含日期、候选解释、采用结果和精确来源位置。改变既有规范音素属于可审计的规则版本变更。
