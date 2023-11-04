# NLP菜鸟逆袭记

- LLMs 千面郎君：https://github.com/km1994/LLMs_interview_notes
  - 介绍：该仓库主要记录 大模型（LLMs） 算法工程师相关的面试题
- LLMs九层妖塔：https://github.com/km1994/LLMsNineStoryDemonTower
  - 介绍：【LLMs九层妖塔】分享 LLMs在自然语言处理（ChatGLM、Chinese-LLaMA-Alpaca、小羊驼 Vicuna、LLaMA、GPT4ALL等）、信息检索（langchain）、语言合成、语言识别、多模态等领域（Stable Diffusion、MiniGPT-4、VisualGLM-6B、Ziya-Visual等）等 实战与经验。
- NLP菜鸟逆袭记：https://github.com/km1994/AwesomeNLP
  - 介绍：【NLP菜鸟逆袭】分享 自然语言处理（文本分类、信息抽取、知识图谱、机器翻译、问答系统、文本生成、Text-to-SQL、文本纠错、文本挖掘、知识蒸馏、模型加速、OCR、TTS、Prompt、embedding等）等 实战与经验。
- NLP 面无不过：https://github.com/km1994/NLP-Interview-Notes
  - 介绍：该仓库主要记录 NLP 算法工程师相关的面试题
- 【关于 NLP】 那些你不知道的事：https://github.com/km1994/nlp_paper_study
  - 介绍：该仓库主要记录 NLP 算法工程师相关的顶会论文研读笔记

梳理 NLP基础任务（文本分类、命名实体识别、关系抽取、事件抽取、文本摘要、文本生成、Prompt）和 LLMs 大模型等开源项目，争取做成一个全网最全NLP小白入门教程！

- [NLP菜鸟逆袭记](#nlp菜鸟逆袭记)
  - [一、文本分类](#一文本分类)
    - [1.1 多类别文本分类](#11-多类别文本分类)
    - [1.2 多标签文本分类](#12-多标签文本分类)
    - [1.3 方面级情感识别](#13-方面级情感识别)
    - [1.4 文本匹配](#14-文本匹配)
  - [二、信息抽取](#二信息抽取)
    - [2.1 命名实体识别](#21-命名实体识别)
    - [2.2 关系抽取](#22-关系抽取)
    - [2.3 事件抽取](#23-事件抽取)
    - [2.4 属性抽取](#24-属性抽取)
    - [2.5 关键词抽取](#25-关键词抽取)
    - [2.6 新词发现](#26-新词发现)
  - [三、知识图谱](#三知识图谱)
    - [3.1 知识图谱](#31-知识图谱)
    - [3.2 实体链指](#32-实体链指)
    - [3.3 知识图谱补全](#33-知识图谱补全)
    - [3.4 neo4j](#34-neo4j)
  - [四、机器翻译](#四机器翻译)
  - [五、问答系统](#五问答系统)
    - [5.1 阅读理解](#51-阅读理解)
    - [5.2 检索式问答](#52-检索式问答)
    - [5.3 基于知识图谱问答](#53-基于知识图谱问答)
    - [5.4 基于知识图谱问答](#54-基于知识图谱问答)
  - [六、文本生成](#六文本生成)
  - [七、Text-to-SQL](#七text-to-sql)
  - [八、文本纠错](#八文本纠错)
  - [九、文本挖掘](#九文本挖掘)
  - [十、知识蒸馏](#十知识蒸馏)
  - [十一、模型加速](#十一模型加速)
    - [11.1 CTranslate2](#111-ctranslate2)
    - [11.2 optimum](#112-optimum)
  - [十二、OCR](#十二ocr)
    - [12.1 pytesseract](#121-pytesseract)
    - [12.2 hn\_ocr](#122-hn_ocr)
    - [12.3 PaddleOCR](#123-paddleocr)
  - [十三、TTS](#十三tts)
    - [13.1 pyttsx3](#131-pyttsx3)
    - [13.2 PaddleSpeech](#132-paddlespeech)
    - [13.3 tensorflow\_tts](#133-tensorflow_tts)
    - [13.4 KAN\_TTS](#134-kan_tts)
  - [十四、Prompt](#十四prompt)
  - [十五、embedding](#十五embedding)
  - [NLP 神器](#nlp-神器)

## 一、文本分类

### 1.1 多类别文本分类

- [NLP菜鸟逆袭记——【多类别文本分类】笔记](https://articles.zsxq.com/id_vblujnqj3vq8.html)
- 多类别文本分类 实战篇
  - [NLP菜鸟逆袭记——【多类别文本分类】实战](https://articles.zsxq.com/id_veej728vewpa.html)
    - 非预训练类模型
      - FastText
      - TextCNN
      - TextRNN
      - TextRCNN
      - Transformer
    - 预训练类模型
      - Bert
      - Albert
      - Roberta
      - Distilbert
      - Electra

### 1.2 多标签文本分类

- [NLP菜鸟逆袭记——【多标签文本分类】笔记](https://articles.zsxq.com/id_5koz88w4spzg.html)
- 多标签文本分类 实战篇
  - [NLP菜鸟逆袭记——【基于 Bert 中文多标签分类】实战](https://articles.zsxq.com/id_23szh9e03eg4.html)
  - [NLP菜鸟逆袭记——【剧本角色情感 中文多标签分类】实战](https://articles.zsxq.com/id_l2f8x5cdt77c.html)

### 1.3 方面级情感识别

- [NLP菜鸟逆袭记——【基于方面的情感分析(ABSA)】理论](https://articles.zsxq.com/id_miwvngdlw5cs.html)
- 基于方面的情感分析(ABSA) 实战篇
  - [NLP菜鸟逆袭记——【基于 Bert 中文方面级情感识别】实战](https://articles.zsxq.com/id_a9lfu00i9w54.html)

### 1.4 文本匹配

- [NLP菜鸟逆袭记——【文本匹配】理论](https://articles.zsxq.com/id_nzhuwrvvvx30.html)
- 文本匹配 实战篇
  - [NLP菜鸟逆袭记——【文本匹配】实战](https://articles.zsxq.com/id_on2rop7drwpb.html)

## 二、信息抽取

### 2.1 命名实体识别

- 命名实体识别 理论篇
  - [NLP菜鸟逆袭记——【HMM->MEMM->CRF】实战](https://articles.zsxq.com/id_50p300pz7oms.html)
  - [DNN-CRF 理论篇](https://articles.zsxq.com/id_on8p4e823wob.html)
- 命名实体识别 实战篇
  - [NLP菜鸟逆袭记——【Bert-CRF】实战](https://articles.zsxq.com/id_2w2wvlnlsl9e.html)
  - [NLP菜鸟逆袭记——【Bert-Softmax】实战](https://articles.zsxq.com/id_93itkfspahti.html)
  - [NLP菜鸟逆袭记——【Bert-Span】实战](https://articles.zsxq.com/id_uubz19x6zqkk.html)
  - [NLP菜鸟逆袭记——【MRC for Flat Nested NER：一种基于机器阅读理解的命名实体识别】实战](https://articles.zsxq.com/id_5eb5kutqpkkx.html)
  - [NLP菜鸟逆袭记——【Biaffine NER：一种基于双仿射注意力机制的命名实体识别】实战](https://articles.zsxq.com/id_jma1qe0cfpru.html)
  - [NLP菜鸟逆袭记——【Multi Head Selection Ner： 一种基于多头选择的命名实体识别】实战](https://articles.zsxq.com/id_nwapunoqma83.html)
  - [NLP菜鸟逆袭记——【one vs rest NER： 一种基于one vs rest的命名实体识别】实战](https://articles.zsxq.com/id_10adh3w5lf43.html)
  - [NLP菜鸟逆袭记——【GlobalPointer：一种基于span分类的解码方法】实战](https://articles.zsxq.com/id_1qzdewzrwwcv.html)
  - [NLP菜鸟逆袭记——【W2NER：一种统一的命名实体识别词与词的的命名实体识别】实战](https://articles.zsxq.com/id_3wkwpcqzog06.html)

### 2.2 关系抽取

- [NLP菜鸟逆袭记——【关系抽取（分类）】理论](https://articles.zsxq.com/id_bko5bhw4wp0g.html)
- 关系抽取 实战篇
  - [NLP菜鸟逆袭记——【BERT-RE：一种基于 Bert 的 Pipeline 实体关系抽取】实践](https://articles.zsxq.com/id_8pvqg3sbpd6x.html)
  - [NLP菜鸟逆袭记——【Casrel Triple Extraction：一种基于 CasRel 的 三元组抽取】实践](https://articles.zsxq.com/id_vac8j1kidxw2.html)
  - [NLP菜鸟逆袭记——【GPLinker：一种基于 GPLinker的 三元组抽取】实践](https://articles.zsxq.com/id_okpfgpofwrib.html)

### 2.3 事件抽取

- 事件抽取 理论篇
- 事件抽取 实战篇
  - [ NLP菜鸟逆袭记——【BERT Event Extraction：一种基于 Bert 的 Pipeline 事件抽取】实践](https://articles.zsxq.com/id_s1mrunww6el4.html)
  - [NLP菜鸟逆袭记——【BERT MRC Event Extraction：一种基于 MRC 的 事件抽取】实践](https://articles.zsxq.com/id_qfai1ixcoogi.html)

### 2.4 属性抽取

- [NLP菜鸟逆袭记——【属性抽取（Attribute Extraction）】理论](https://articles.zsxq.com/id_t6zkk3oolgcb.html)
- 属性抽取 实战篇
  - [NLP菜鸟逆袭记——【一种基于 albert 的中文属性抽取 —— Albert for Attribute Extraction】实践](https://articles.zsxq.com/id_rbkjlutnsuhu.html)

### 2.5 关键词抽取

- [【NLP菜鸟逆袭记—【关键词提取】理论](https://articles.zsxq.com/id_igmn1m26r4si.html)
- 关键词抽取 实战篇

### 2.6 新词发现

- [NLP菜鸟逆袭记—【新词发现】理论](https://articles.zsxq.com/id_qb0c3wuvj7sk.html)
- 新词发现 实战篇

## 三、知识图谱

### 3.1 知识图谱

- [【NLP菜鸟逆袭记—【知识图谱】理论](https://articles.zsxq.com/id_tw83e60ocdw0.html)
- 知识图谱 实战篇
  - [NLP菜鸟逆袭记—【基于金融知识图谱的知识计算引擎构建】实战](https://articles.zsxq.com/id_a6hrj3a58h3f.html)
  - [NLP菜鸟逆袭记—【基于金融知识图谱的问答系统】实战](https://articles.zsxq.com/id_8b84m3blgq6d.html)

### 3.2 实体链指

- [【NLP菜鸟逆袭记—【实体链指】理论](https://articles.zsxq.com/id_2r1mf9a5p3vg.html)
- 实体链指 实战篇

### 3.3 知识图谱补全

- [【NLP菜鸟逆袭记—【知识图谱补全】理论](https://articles.zsxq.com/id_izt2xkwxgtif.html)
- 知识图谱补全 实战篇

### 3.4 neo4j

- [【NLP菜鸟逆袭记—【Neo4j】实战](https://articles.zsxq.com/id_z9wen0ursw6i.html)

## 四、机器翻译

- [NLP菜鸟逆袭记—【机器翻译】理论](https://articles.zsxq.com/id_6w5qr770n5j8.html)
- 机器翻译 实战篇
  - [NLP菜鸟逆袭记—【seq2seq_english_to_chinese 一种结合 seq2seq 的 文本翻译】理论](https://articles.zsxq.com/id_c9pxfdewm4e8.html)

## 五、问答系统

- [NLP菜鸟逆袭记—【智能问答技术】理论](https://articles.zsxq.com/id_kahbqgjn1wh8.html)

### 5.1 阅读理解

- [NLP菜鸟逆袭记—【机器阅读理解】理论](https://articles.zsxq.com/id_xjrvml06w25j.html)
- 阅读理解 实战篇
  - [NLP菜鸟逆袭记—【基于QANet的中文阅读理解】实战](https://articles.zsxq.com/id_8djhwl6wwnb4.html)

### 5.2 检索式问答

- [NLP菜鸟逆袭记—【FAQ 检索式问答系统】理论](https://articles.zsxq.com/id_ujnvmd5j5vza.html)
- 检索式问答 实战篇
  - [NLP菜鸟逆袭记—【Faiss】实践](https://articles.zsxq.com/id_jk7hbbp344fc.html)
  - [NLP菜鸟逆袭记—【milvus】理论](https://articles.zsxq.com/id_kg8ba02hwhjb.html)

### 5.3 基于知识图谱问答

- [NLP菜鸟逆袭记—【KBQA】理论](https://articles.zsxq.com/id_2dxaer57pdv7.html)
- 基于知识图谱问答 实战篇
  - [NLP菜鸟逆袭记—【基于金融知识图谱的知识计算引擎构建】实战](https://articles.zsxq.com/id_a6hrj3a58h3f.html)
  - [NLP菜鸟逆袭记—【基于金融知识图谱的问答系统】实战](https://articles.zsxq.com/id_8b84m3blgq6d.html)

### 5.4 基于知识图谱问答

- [NLP菜鸟逆袭记—【对话系统】理论](https://articles.zsxq.com/id_p80zsawpxh1e.html)
- 对话系统 实战篇

## 六、文本生成

- [NLP菜鸟逆袭记—【自然语言生成】理论](https://articles.zsxq.com/id_spc0v7gmqx7r.html)
- 文本生成 实战篇
  - [NLP菜鸟逆袭记—【Bert_Unilm】实践](https://articles.zsxq.com/id_q1j15rvqlf1r.html)
  - [NLP菜鸟逆袭记—【T5_Pegasus】实践](https://articles.zsxq.com/id_v1getivgjmdg.html)

## 七、Text-to-SQL

- [NLP菜鸟逆袭记—【Text-to-SQL】理论](https://articles.zsxq.com/id_wmwrp16p0wjh.html)
- Text-to-SQL 实战篇

## 八、文本纠错

- [NLP菜鸟逆袭记—【文本纠错】理论](https://articles.zsxq.com/id_j40kicdoi4su.html)
- 文本纠错 实战篇
  - [NLP菜鸟逆袭记—【一种结合 Bert 的 中文拼写检查】实战](https://articles.zsxq.com/id_wi45ubrg8xsm.html)
  - [NLP菜鸟逆袭记—【CSC 一种结合 Soft-Masked Bert 的 中文拼写检查】实战](https://articles.zsxq.com/id_snw6lmzidcgw.html)

## 九、文本挖掘

- [NLP菜鸟逆袭记—【文本挖掘】理论](https://articles.zsxq.com/id_g3qujbn4slba.html)
- 文本挖掘 实战篇

## 十、知识蒸馏

- [NLP菜鸟逆袭记—【Bert 压缩】理论](https://articles.zsxq.com/id_bxue7x82vew1.html)
  - [NLP菜鸟逆袭记【FastBERT】理论](https://articles.zsxq.com/id_q69tgj5gdo86.html)
- 知识蒸馏 实战篇
  - [NLP菜鸟逆袭记【Distilling Task-Specific from BERT into SNN】实战](https://articles.zsxq.com/id_1jvhfo38j70j.html)
  - [NLP菜鸟逆袭记【FastBERT】实战](https://articles.zsxq.com/id_8qcmepswwd2x.html)

## 十一、模型加速

### 11.1 CTranslate2

- [NLP菜鸟逆袭记—【模型加速 —— CTranslate2】理论](https://articles.zsxq.com/id_u9jnt7p9fm0l.html)

### 11.2 optimum

- [NLP菜鸟逆袭记—【模型加速 —— Optimum】理论](https://articles.zsxq.com/id_g7jbion16zch.html)

## 十二、OCR

- [NLP菜鸟逆袭记—【OCR】理论](https://articles.zsxq.com/id_mqluwdt9rrf7.html)

### 12.1 pytesseract

- [NLP菜鸟逆袭记—【OCR —— tesseract】理论](https://articles.zsxq.com/id_ska7wj60pz5m.html)

### 12.2 hn_ocr

- [NLP菜鸟逆袭记—【OCR —— hn_ocr】理论](https://articles.zsxq.com/id_kqqb39lvfawx.html)

### 12.3 PaddleOCR

- [NLP菜鸟逆袭记—【OCR —— PaddleOCR】理论](https://articles.zsxq.com/id_c7b8pp3fg0zn.html)

## 十三、TTS

- [NLP菜鸟逆袭记—【文本语音合成 TTS】理论](https://articles.zsxq.com/id_ev7d5hw63spx.html)

### 13.1 pyttsx3

- [NLP菜鸟逆袭记—【文本语音合成 —— pyttsx3】实战](https://articles.zsxq.com/id_uebr3ic72dg9.html)

### 13.2 PaddleSpeech

- PaddleSpeech 理论篇

### 13.3 tensorflow_tts

- [NLP菜鸟逆袭记—【文本语音合成 —— tensorflow_tts】实战](https://articles.zsxq.com/id_j1h6wop2zjqn.html)

### 13.4 KAN_TTS

- [NLP菜鸟逆袭记—【文本语音合成 —— KAN-TTS】实战](https://articles.zsxq.com/id_8jx9j9gojwwq.html)

## 十四、Prompt

- NLP菜鸟逆袭记—【Prompt】实战
- Prompt 实战篇
  - [NLP菜鸟逆袭记—【PromptCLUE】实战](https://articles.zsxq.com/id_a2hmp2u76430.html)

## 十五、embedding

- [NLP菜鸟逆袭记—【Embeddings】理论](https://articles.zsxq.com/id_zurkdiso7had.html)
- embedding 实战篇
  - [NLP菜鸟逆袭记—【sbert】实战](https://articles.zsxq.com/id_4murr07k07is.html)
  - [NLP菜鸟逆袭记—【text2vec】实战](https://articles.zsxq.com/id_clin3x4nnwb3.html)
  - [NLP菜鸟逆袭记—【SGPT:基于GPT的生成式embedding】实战](https://articles.zsxq.com/id_ba64rx7dcejw.html)
  - [NLP菜鸟逆袭记—【BGE —— 智源开源最强语义向量模型】实战](https://articles.zsxq.com/id_j2k2i3efzqal.html)
  - [NLP菜鸟逆袭记—【M3E：一种大规模混合embedding】实战](https://articles.zsxq.com/id_sqx0uhtzhzrq.html)

## NLP 神器

- [chaizi：一种 汉语拆字词典 神器](https://articles.zsxq.com/id_quxhoblg3mqm.html)
- [cn2an：一种中文数字与阿拉伯数字的相互转换神器](https://articles.zsxq.com/id_xtz9lhwmwu4i.html)
- [cocoNLP：一种 人名、地址、邮箱、手机号、手机归属地 等信息的抽取，rake短语抽取算法](https://articles.zsxq.com/id_ehituke9kmcp.html)
- [difflib.SequenceMatcher：一种 文本查重 神器](https://articles.zsxq.com/id_rworrt2itbx0.html)
- [Entity_Emotion_Express：一种 词汇情感值 神器](https://articles.zsxq.com/id_5wd7qs2s7wuc.html)
- [jieba_fast：一种 中文分词 神器](https://articles.zsxq.com/id_ce1evmm35eca.html)
- [JioNLP：一种 中文 NLP 预处理 神器](https://articles.zsxq.com/id_77joqloxpf01.html)
- [ngender：一种 根据名字判断性别 神器](https://articles.zsxq.com/id_mwz92b7wzilp.html)
- [pdfplumber：一种 pdf 内容解析神器](https://articles.zsxq.com/id_9ebhgfl8ancz.html)
- [phone：一种 中国手机归属地查询 神器](https://articles.zsxq.com/id_n0fcegesuiij.html)
- [PrettyTable：一种 生成美观的ASCII格式的表格 神器](https://articles.zsxq.com/id_nxzf7p3oxne9.html)
- [Pypinyin：一种汉字转拼音神器](https://articles.zsxq.com/id_qk6s5jqvgai5.html)
- [Rank-BM25：一种 基于bm25算法 神器](https://articles.zsxq.com/id_l97rw5i61cdc.html)
- [schedule ：一种 最全的Python定时任务神器](https://articles.zsxq.com/id_phw1qiptlqnk.html)
- [similarity：一种 相似度计算 神器](https://articles.zsxq.com/id_wpunbmh5sa9w.html)
- [SnowNLP：一种 中文文本预处理 神器](https://articles.zsxq.com/id_zidnh0s32vi1.html)
- [Synonyms：一种中文近义词 神器](https://articles.zsxq.com/id_xalkpstvjubb.html)
- [textfilter：一种 中英文敏感词过滤 神器](https://articles.zsxq.com/id_606jk9v0coh8.html)
- [一种 中文缩写库 神器](https://articles.zsxq.com/id_equvx9vej3xw.html)

