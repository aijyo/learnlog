1. 技术文档(2 个 .md 文件,放在项目根)
文件	内容	体量
README.md	项目导览 + 快速开始 + 端到端流程示例 + 目录结构 + 注意事项	4.9 KB
TECHNICAL_DESIGN.md	完整技术参考——动机/威胁模型 + 架构 + 4 个 demo 深度解析(含字节级原理) + 限制 + 如何应用到你自己的 EXE(3 种场景) + PE 字段速查表 + 哈希算法速查 + imphash 精确定义	31.8 KB
TECHNICAL_DESIGN.md 的 9 个章节:

项目动机和威胁模型(明确说"绕静态指纹不等于绕 EDR")
整体架构(编译期 vs 后处理两条线互补)
demo_iat 深度解析:ROR-13 公式、strenc.h 编译期 XOR 链、resolve.h EAT 走法
patchtool 深度解析:7 类改动的精确 PE 字节位置 + 公式(TimeDateStamp offset、Section table 偏移、Fisher-Yates LCG 参数等)
pediff 深度解析:5 个 tab 各自检测什么、染色规则、imphash 自实现细节
hashlab 深度解析:5 个功能各自的用途和实现
限制和已知问题(诚实列出 5 类做不到的事 + 各 demo 的具体局限)
迁移到你自己的 EXE — 3 种场景的 step-by-step:
场景 A:有源码 → 拷头文件 + 改 APIS[] + 改调用点
场景 B:已编译 EXE → patchtool CLI 单命令
场景 C:混合(推荐生产场景,每个 target 一份不同 seed)
参考:PE 字段速查表 + 6 种哈希算法速查 + imphash 算法精确定义 + LCG 参数
2. 源码 + 文档 zip
D:\tmp\rename_source.zip — 74.9 KB,23 个文件:

rename_source.zip
├── CMakeLists.txt              顶层(/MT + output/ 统一)
├── README.md                   项目导览
├── TECHNICAL_DESIGN.md         完整技术参考
├── _build_all.bat              原 batch 流程(保留)
├── demo_iat/                   7 个文件 (.cpp/.h/.py + CMake + batch)
├── patchtool/                  4 个文件
├── pediff/                     3 个文件
└── hashlab/                    3 个文件
已端到端验证可移植:

解压到全新目录 D:\tmp\rename_dist_test\
cmake -B build -G "Visual Studio 17 2022" -A x64 → 干净通过(6.7s)
cmake --build build --config Release --target rename_all → 零警告零错误
POST_BUILD 自动跑 [PASS] target APIs not in IAT
6 个 EXE 全部生成到 output/
在另一台机器上用
# 解压到任意位置
Expand-Archive rename_source.zip -DestinationPath C:\work\rename
cd C:\work\rename

# 一次配置,常用 build
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target rename_all

# 6 个 EXE 自动出现在 .\output\
先决条件(已写在 README 里):VS 2022 Community C++ workload + Python 3.7+ + CMake 3.20+。

如果要分发只给已编译产物,把 output/ 单独 zip 起来即可(全部静态链接,纯系统 DLL 依赖)。
